"""
Playwright browser manager with helper utilities for the form-fill agent.
"""
import asyncio
from typing import Optional
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Playwright


class BrowserManager:
    """Manages a Playwright browser instance for the agent."""

    def __init__(self):
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self):
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
            locale="en-IN",
        )
        self.page = await self._context.new_page()
        # Block images/fonts for speed (we still take screenshots)
        # Only block tracking/ads
        await self._context.route(
            "**/*.{woff,woff2,ttf}",
            lambda route: route.abort(),
        )

    async def stop(self):
        try:
            if self.page:
                await self.page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass

    async def screenshot_bytes(self) -> bytes:
        """Take a full-page screenshot and return as bytes."""
        return await self.page.screenshot(full_page=True, type="png")

    async def get_page_state(self) -> dict:
        """Extract current page state for the LLM planner."""
        try:
            title = await self.page.title()
            url = self.page.url

            # Get all interactive elements
            form_fields = await self.page.evaluate("""
                () => Array.from(
                    document.querySelectorAll('input, select, textarea, button[type="submit"], button[type="button"]')
                ).map(el => ({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    name: el.name || null,
                    id: el.id || null,
                    placeholder: el.placeholder || null,
                    label: el.labels && el.labels[0] ? el.labels[0].innerText.trim() : null,
                    value: el.value || null,
                    required: el.required || false,
                    disabled: el.disabled || false,
                    visible: el.offsetParent !== null,
                    className: el.className ? el.className.substring(0, 60) : null,
                })).filter(el => el.visible)
            """)

            # Get visible text (truncated to save tokens)
            visible_text = await self.page.evaluate("""
                () => {
                    const el = document.body.cloneNode(true);
                    // Remove script/style elements
                    el.querySelectorAll('script, style, nav, footer').forEach(e => e.remove());
                    return el.innerText.replace(/\\s+/g, ' ').substring(0, 3000);
                }
            """)

            # Detect error messages
            error_indicators = await self.page.evaluate("""
                () => {
                    const errorEls = document.querySelectorAll(
                        '.error, .alert-danger, .error-message, [class*="error"], [class*="invalid"]'
                    );
                    return Array.from(errorEls).map(e => e.innerText.trim()).filter(t => t.length > 0);
                }
            """)

            return {
                "url": url,
                "title": title,
                "visible_text": visible_text,
                "form_fields": form_fields,
                "error_indicators": error_indicators,
            }
        except Exception as e:
            return {"url": self.page.url, "error": str(e), "form_fields": [], "visible_text": ""}

    async def safe_click(self, selector: str) -> bool:
        """Try multiple selector strategies to click an element."""
        strategies = [
            lambda: self.page.click(selector, timeout=5000),
            lambda: self.page.click(f"text={selector}", timeout=5000),
            lambda: self.page.locator(selector).first.click(timeout=5000),
        ]
        for strategy in strategies:
            try:
                await strategy()
                await self.page.wait_for_load_state("networkidle", timeout=10000)
                return True
            except Exception:
                continue
        return False

    async def safe_fill(self, selector: str, value: str) -> bool:
        """Try to fill a field with multiple selector strategies."""
        try:
            # First try direct selector
            el = self.page.locator(selector).first
            await el.wait_for(state="visible", timeout=5000)
            await el.clear()
            await el.fill(value)
            return True
        except Exception:
            pass

        # Try by label text
        try:
            el = self.page.get_by_label(selector)
            await el.fill(value)
            return True
        except Exception:
            pass

        # Try by placeholder
        try:
            el = self.page.get_by_placeholder(selector)
            await el.fill(value)
            return True
        except Exception:
            pass

        return False

    async def safe_select(self, selector: str, value: str) -> bool:
        """Select a dropdown option."""
        try:
            await self.page.select_option(selector, label=value, timeout=5000)
            return True
        except Exception:
            pass
        try:
            await self.page.select_option(selector, value=value, timeout=5000)
            return True
        except Exception:
            pass
        # Try clicking the option text directly (custom dropdowns)
        try:
            await self.page.click(selector)
            await asyncio.sleep(0.5)
            await self.page.click(f"text={value}")
            return True
        except Exception:
            pass
        return False

    async def upload_file(self, selector: str, file_path: str) -> bool:
        """Trigger file upload on an input[type=file]."""
        try:
            async with self.page.expect_file_chooser() as fc_info:
                await self.page.click(selector)
            file_chooser = await fc_info.value
            await file_chooser.set_files(file_path)
            return True
        except Exception:
            try:
                await self.page.set_input_files(selector, file_path)
                return True
            except Exception:
                return False
