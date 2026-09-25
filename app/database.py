from supabase import create_client, Client
from app.config import settings

_supabase_client: Client | None = None


def get_supabase() -> Client:
    """Return a Supabase client using the service-role key (server-side)."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _supabase_client


def get_supabase_anon() -> Client:
    """Return a Supabase client using the anon key (respects RLS)."""
    return create_client(settings.supabase_url, settings.supabase_anon_key)
