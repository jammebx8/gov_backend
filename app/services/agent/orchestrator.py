"""
Core agentic orchestrator: ReAct loop that drives the form-fill agent.

Flow per iteration:
  1. Take screenshot
  2. Get page state (form fields, visible text)
  3. Call Grok planner → get next action
  4. Execute action via Playwright
  5. Log step + upload screenshot
  6. Repeat until done/failed/needs_human
"""
import asyncio
import base64
import json
import tempfile
import os
from datetime import datetime, timezone
from typing import Optional
from app.services.agent.browser import BrowserManager
from app.services.agent.planner import plan_next_action
from app.services.agent.verifier import verify_form
from app.utils.storage import upload_screenshot
from app.database import get_supabase
from app.config import settings


MAX_ITERATIONS = 35
STEP_DELAY_MS = 1500  # wait between actions to let pages settle


class FormFillAgent:
    def __init__(self, job_id: str, user: dict, scheme: dict, documents: list[dict]):
        self.job_id = job_id
        self.user = user
        self.scheme = scheme
        self.documents = documents
        self.steps: list[dict] = []
        self.screenshot_urls: list[str] = []
        self.db = get_supabase()
        self.browser = BrowserManager()
        self.user_data = self._build_user_data_map()

    def _build_user_data_map(self) -> dict:
        """
        Flatten all user profile fields + extracted document data into one dict
        that the LLM can reference to fill fields.
        """
        data = {
            "full_name": self.user.get("full_name", ""),
            "email": self.user.get("email", ""),
            "phone": self.user.get("phone", ""),
            "date_of_birth": str(self.user.get("date_of_birth", "")),
            "gender": self.user.get("gender", ""),
            "state": self.user.get("state", ""),
            "district": self.user.get("district", ""),
            "annual_income": self.user.get("annual_income", ""),
            "caste_category": self.user.get("caste_category", ""),
            "occupation": self.user.get("occupation", ""),
            "is_student": self.user.get("is_student", False),
            "is_farmer": self.user.get("is_farmer", False),
            "disability_status": self.user.get("disability_status", False),
        }

        # Merge in extracted document fields (documents take precedence for official names)
        for doc in self.documents:
            extracted = doc.get("extracted_data") or {}
            doc_type = doc.get("doc_type", "")
            if isinstance(extracted, dict) and not extracted.get("parse_error"):
                for key, value in extracted.items():
                    # Prefix with doc type to avoid collisions
                    data[f"{doc_type}_{key}"] = value
                    # For common fields, also store at top level if not already set
                    if key in ("name", "date_of_birth", "gender") and not data.get(key):
                        data[key] = value
                    if key == "aadhaar_number":
                        data["aadhaar_number"] = value
                    if key == "pan_number":
                        data["pan_number"] = value
                    if key == "account_number":
                        data["bank_account_number"] = value
                    if key == "ifsc_code":
                        data["ifsc_code"] = value
                    if key == "annual_income" and not data.get("annual_income"):
                        data["annual_income"] = value

        # Map document file URLs for uploads
        for doc in self.documents:
            if doc.get("file_url"):
                data[f"file_url_{doc['doc_type']}"] = doc["file_url"]

        return data

    async def _log_step(
        self,
        action_type: str,
        description: str,
        screenshot_url: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        selector: Optional[str] = None,
        value: Optional[str] = None,
    ):
        step = {
            "step": len(self.steps) + 1,
            "action_type": action_type,
            "description": description,
            "selector": selector,
            "value": value,
            "screenshot_url": screenshot_url,
            "success": success,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.steps.append(step)

        # Live-update the job in DB
        try:
            self.db.table("application_jobs").update({
                "agent_log": self.steps,
                "screenshot_urls": self.screenshot_urls,
                "status": "running",
            }).eq("id", self.job_id).execute()
        except Exception:
            pass

    async def _take_screenshot_and_upload(self, iteration: int) -> tuple[bytes, str, str]:
        """Take screenshot, upload it, return (bytes, b64, url)."""
        screenshot_bytes = await self.browser.screenshot_bytes()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

        try:
            url = await upload_screenshot(screenshot_bytes, self.job_id, iteration)
            self.screenshot_urls.append(url)
        except Exception:
            url = ""

        return screenshot_bytes, screenshot_b64, url

    async def _execute_action(self, action: dict) -> tuple[bool, Optional[str]]:
        """
        Execute a planned action. Returns (success, error_message).
        """
        action_type = action.get("type")

        if action_type == "fill":
            selector = action.get("selector", "")
            value = action.get("value", "")
            if not value:
                return True, None  # skip empty fills
            success = await self.browser.safe_fill(selector, str(value))
            await asyncio.sleep(0.3)
            return success, None if success else f"Could not fill: {selector}"

        elif action_type == "click":
            selector = action.get("selector", "")
            success = await self.browser.safe_click(selector)
            await asyncio.sleep(STEP_DELAY_MS / 1000)
            return success, None if success else f"Could not click: {selector}"

        elif action_type == "select":
            selector = action.get("selector", "")
            value = action.get("value", "")
            success = await self.browser.safe_select(selector, str(value))
            await asyncio.sleep(0.3)
            return success, None if success else f"Could not select: {selector}"

        elif action_type == "upload":
            doc_type = action.get("doc_type", "")
            selector = action.get("selector", "input[type='file']")
            # Find the document file URL
            file_url_key = f"file_url_{doc_type}"
            file_url = self.user_data.get(file_url_key)
            if not file_url:
                return False, f"No document available for type: {doc_type}"

            # Download the file temporarily
            import httpx
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(file_url)
                    file_bytes = resp.content

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg" if file_url.endswith((".jpg", ".jpeg")) else ".pdf"
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                success = await self.browser.upload_file(selector, tmp_path)
                os.unlink(tmp_path)
                return success, None if success else f"Could not upload to: {selector}"
            except Exception as e:
                return False, f"Upload error: {str(e)}"

        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = 600 if direction == "down" else -600
            await self.browser.page.evaluate(f"window.scrollBy(0, {amount})")
            await asyncio.sleep(0.5)
            return True, None

        elif action_type == "wait":
            ms = action.get("ms", 2000)
            await asyncio.sleep(ms / 1000)
            return True, None

        elif action_type in ("verify", "submit", "done", "needs_human"):
            # These are handled by the main loop
            return True, None

        return False, f"Unknown action type: {action_type}"

    async def run(self):
        """Main agent execution loop."""
        try:
            # Mark job as running
            self.db.table("application_jobs").update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", self.job_id).execute()

            await self.browser.start()
            application_url = self.scheme.get("application", "")

            if not application_url or application_url.strip() == "":
                raise ValueError("No application URL found for this scheme")

            await self.browser.page.goto(application_url, wait_until="networkidle", timeout=30000)
            await self._log_step("navigate", f"Navigated to {application_url}")

            verify_done = False
            consecutive_failures = 0

            for iteration in range(MAX_ITERATIONS):
                await asyncio.sleep(STEP_DELAY_MS / 1000)

                # Take screenshot and get page state
                _, screenshot_b64, screenshot_url = await self._take_screenshot_and_upload(iteration)
                page_state = await self.browser.get_page_state()

                # Plan next action
                try:
                    action = await plan_next_action(
                        page_state=page_state,
                        screenshot_b64=screenshot_b64,
                        user_data=self.user_data,
                        scheme_name=self.scheme.get("scheme_name", ""),
                        previous_steps=self.steps,
                    )
                except Exception as e:
                    await self._log_step("plan_error", f"Planner error: {str(e)}", success=False, error=str(e))
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError(f"Planner failed 3 times: {e}")
                    continue

                action_type = action.get("type")
                description = action.get("description") or action.get("field_name") or action_type

                await self._log_step(
                    action_type=action_type,
                    description=f"[{action_type}] {description}",
                    screenshot_url=screenshot_url,
                    selector=action.get("selector"),
                    value=action.get("value"),
                )

                # Handle terminal states
                if action_type == "done":
                    ref_id = action.get("reference_number") or action.get("ref_id", "")
                    await self._complete(ref_id, screenshot_url)
                    return

                if action_type == "needs_human":
                    reason = action.get("reason", "Manual intervention required")
                    await self._pause_for_human(reason, screenshot_url)
                    return

                # Verification step
                if action_type == "verify" and not verify_done:
                    verification = await verify_form(
                        screenshot_b64=screenshot_b64,
                        user_data=self.user_data,
                        scheme_name=self.scheme.get("scheme_name", ""),
                    )
                    verify_done = True

                    if verification.get("ready_to_submit"):
                        await self._log_step("verify_passed", "Form verification passed — ready to submit")
                    else:
                        issues = verification.get("issues", [])
                        await self._log_step(
                            "verify_failed",
                            f"Verification issues: {'; '.join(issues)}",
                            success=False,
                        )
                        # Reset verify_done so we can verify again after corrections
                        verify_done = False
                    continue

                # Submit action
                if action_type == "submit":
                    selector = action.get("selector", "button[type='submit']")
                    success = await self.browser.safe_click(selector)
                    if success:
                        await asyncio.sleep(3)  # wait for confirmation page
                        _, screenshot_b64_post, screenshot_url_post = await self._take_screenshot_and_upload(iteration + 100)
                        # Check if submission succeeded
                        page_state_post = await self.browser.get_page_state()
                        text = page_state_post.get("visible_text", "").lower()
                        if any(kw in text for kw in ["success", "submitted", "application number", "reference", "thank you", "congratulations"]):
                            # Extract reference number
                            import re
                            ref_match = re.search(r"(?:application|reference|app|reg)[^\d]*(\d{5,20})", text, re.IGNORECASE)
                            ref_id = ref_match.group(1) if ref_match else ""
                            await self._complete(ref_id, screenshot_url_post)
                            return
                    consecutive_failures += 1
                    continue

                # Execute the action
                success, error = await self._execute_action(action)
                if not success:
                    consecutive_failures += 1
                    await self._log_step(
                        "action_failed",
                        f"Action failed: {error}",
                        success=False,
                        error=error,
                    )
                    if consecutive_failures >= 5:
                        raise RuntimeError(f"Too many consecutive failures. Last error: {error}")
                else:
                    consecutive_failures = 0

            # Max iterations reached
            raise RuntimeError(f"Max iterations ({MAX_ITERATIONS}) reached without completion")

        except Exception as e:
            await self._fail(str(e))
        finally:
            await self.browser.stop()

    async def _complete(self, ref_id: str, screenshot_url: str = ""):
        await self._log_step("completed", f"Application submitted successfully. Reference: {ref_id}", screenshot_url=screenshot_url)
        self.db.table("application_jobs").update({
            "status": "completed",
            "application_ref_id": ref_id,
            "agent_log": self.steps,
            "screenshot_urls": self.screenshot_urls,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", self.job_id).execute()

    async def _pause_for_human(self, reason: str, screenshot_url: str = ""):
        await self._log_step("needs_human", f"Human intervention needed: {reason}", screenshot_url=screenshot_url)
        self.db.table("application_jobs").update({
            "status": "needs_review",
            "error_details": reason,
            "agent_log": self.steps,
            "screenshot_urls": self.screenshot_urls,
        }).eq("id", self.job_id).execute()

    async def _fail(self, error: str):
        await self._log_step("failed", f"Agent failed: {error}", success=False, error=error)
        self.db.table("application_jobs").update({
            "status": "failed",
            "error_details": error,
            "agent_log": self.steps,
            "screenshot_urls": self.screenshot_urls,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", self.job_id).execute()


async def run_agent(job_id: str, user_id: str, scheme_id: str):
    """Entry point for background task execution."""
    db = get_supabase()

    # Load user
    user_result = db.table("users").select("*").eq("id", user_id).single().execute()
    if not user_result.data:
        db.table("application_jobs").update({"status": "failed", "error_details": "User not found"}).eq("id", job_id).execute()
        return
    user = user_result.data

    # Load scheme
    scheme_result = db.table("government_schemes").select("*").eq("id", scheme_id).single().execute()
    if not scheme_result.data:
        db.table("application_jobs").update({"status": "failed", "error_details": "Scheme not found"}).eq("id", job_id).execute()
        return
    scheme = scheme_result.data

    # Load documents
    docs_result = db.table("user_documents").select("*").eq("user_id", user_id).execute()
    documents = docs_result.data or []

    agent = FormFillAgent(job_id=job_id, user=user, scheme=scheme, documents=documents)
    await agent.run()
