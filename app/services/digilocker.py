"""
DigiLocker OAuth 2.0 integration.
Allows users to connect their DigiLocker account and pull documents directly.
"""
import httpx
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from typing import Optional
from app.config import settings


DIGILOCKER_AUTH_URL = "https://api.digitallocker.gov.in/public/oauth2/1/authorize"
DIGILOCKER_TOKEN_URL = "https://api.digitallocker.gov.in/public/oauth2/1/token"
DIGILOCKER_FILES_URL = "https://api.digitallocker.gov.in/public/oauth2/1/files"
DIGILOCKER_AADHAAR_URL = "https://api.digitallocker.gov.in/public/oauth2/1/xml/eaadhaar"


def get_auth_url(user_id: str) -> str:
    """Generate DigiLocker OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": settings.digilocker_client_id,
        "redirect_uri": settings.digilocker_redirect_uri,
        "state": user_id,
        "scope": "openid profile",
        "code_challenge_method": "plain",
    }
    return f"{DIGILOCKER_AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for access token."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            DIGILOCKER_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": settings.digilocker_client_id,
                "client_secret": settings.digilocker_client_secret,
                "redirect_uri": settings.digilocker_redirect_uri,
            },
        )
        response.raise_for_status()
        return response.json()


async def fetch_issued_documents(access_token: str) -> list[dict]:
    """Fetch list of issued documents from DigiLocker."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            DIGILOCKER_FILES_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])


async def fetch_aadhaar_data(access_token: str) -> dict:
    """
    Fetch eAadhaar XML from DigiLocker and parse it into structured data.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            DIGILOCKER_AADHAAR_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        xml_content = response.text

    return parse_aadhaar_xml(xml_content)


def parse_aadhaar_xml(xml_content: str) -> dict:
    """Parse Aadhaar XML to extract structured personal data."""
    try:
        root = ET.fromstring(xml_content)
        # Aadhaar XML typically has a KycRes/UidData/Poi element
        poi = root.find(".//Poi")
        poa = root.find(".//Poa")

        data = {}
        if poi is not None:
            data.update({
                "name": poi.get("name", ""),
                "date_of_birth": poi.get("dob", ""),
                "gender": poi.get("gender", ""),
                "phone": poi.get("phone", ""),
                "email": poi.get("email", ""),
            })
        if poa is not None:
            data.update({
                "address": poa.get("house", "") + " " + poa.get("loc", ""),
                "district": poa.get("dist", ""),
                "state": poa.get("state", ""),
                "pincode": poa.get("pc", ""),
                "country": poa.get("country", "India"),
            })

        return data
    except ET.ParseError:
        return {"raw_xml": xml_content[:500], "parse_error": True}


async def fetch_document_pdf(access_token: str, uri: str) -> bytes:
    """Fetch a specific document PDF from DigiLocker by URI."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            f"https://api.digitallocker.gov.in/public/oauth2/1/file/{uri}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        return response.content
