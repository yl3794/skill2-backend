"""
Certification verification endpoint.

Certifications are JWT-backed records issued when a worker meets all
certification thresholds for a skill. The JWT is signed with ``CERT_SECRET``
and expires after ``cert_valid_days`` days.

This endpoint is intentionally unauthenticated so that HR systems, third-party
auditors, and compliance platforms can verify certificates without holding an
org API key.
"""

from fastapi import APIRouter

from database import operations as db

router = APIRouter(prefix="/certifications", tags=["Certifications"])


@router.get("/{cert_id}/verify", summary="Verify a certificate of readiness")
async def verify_certification(cert_id: str):
    """Verifies the authenticity and validity of a training certificate.

    Decodes and validates the JWT embedded in the certification record.
    Returns worker identity, skill, and performance metadata so auditors
    can confirm the certificate is genuine and unexpired.

    Args:
        cert_id: UUID of the certification record to verify.

    Returns:
        Dict with ``valid: True`` and certificate details on success, or
        ``valid: False`` with a ``reason`` string on failure (not found,
        expired, or tampered signature).
    """
    return db.verify_certification(cert_id)
