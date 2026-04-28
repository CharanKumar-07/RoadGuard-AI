# backend/alerts/service.py
"""
Alert service: send SMS via Twilio and HTML email via SMTP.

Both transports are optional — if credentials are not configured,
the service logs a warning and skips gracefully.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)


class AlertService:
    """
    Unified alert sender (SMS + email).

    Credentials are read from the environment (set via .env / docker-compose):
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
        SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS
        EMERGENCY_PHONES  (comma-separated)
        EMERGENCY_EMAILS  (comma-separated)
    """

    def __init__(self) -> None:
        # Twilio
        self._twilio_sid    = os.getenv("TWILIO_ACCOUNT_SID",  "")
        self._twilio_token  = os.getenv("TWILIO_AUTH_TOKEN",   "")
        self._twilio_from   = os.getenv("TWILIO_FROM_NUMBER",  "")
        self._twilio_client = None
        if all([self._twilio_sid, self._twilio_token, self._twilio_from]):
            try:
                from twilio.rest import Client
                self._twilio_client = Client(self._twilio_sid, self._twilio_token)
                logger.info("Twilio SMS client initialised.")
            except ImportError:
                logger.warning("twilio package not installed — SMS disabled.")
            except Exception as exc:
                logger.warning("Twilio init failed: %s — SMS disabled.", exc)
        else:
            logger.info("Twilio credentials not configured — SMS disabled.")

        # SMTP
        self._smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self._smtp_port   = int(os.getenv("SMTP_PORT", "587"))
        self._smtp_user   = os.getenv("SMTP_USER",   "")
        self._smtp_pass   = os.getenv("SMTP_PASS",   "")
        self._smtp_ok     = bool(self._smtp_user and self._smtp_pass)
        if not self._smtp_ok:
            logger.info("SMTP credentials not configured — email disabled.")

        # Pre-configured emergency contacts
        self._emergency_phones = [
            p.strip() for p in os.getenv("EMERGENCY_PHONES", "").split(",") if p.strip()
        ]
        self._emergency_emails = [
            e.strip() for e in os.getenv("EMERGENCY_EMAILS", "").split(",") if e.strip()
        ]

    # ------------------------------------------------------------------
    # Low-level senders
    # ------------------------------------------------------------------

    def send_sms(self, to: str, message: str) -> bool:
        """Returns True on success."""
        if not self._twilio_client:
            logger.debug("SMS skipped (Twilio not configured): to=%s", to)
            return False
        try:
            self._twilio_client.messages.create(
                body  = message,
                from_ = self._twilio_from,
                to    = to,
            )
            logger.info("SMS sent to %s", to)
            return True
        except Exception as exc:
            logger.error("SMS failed to %s: %s", to, exc)
            return False

    def send_email(self, to: str, subject: str, body_html: str) -> bool:
        """Returns True on success."""
        if not self._smtp_ok:
            logger.debug("Email skipped (SMTP not configured): to=%s", to)
            return False
        try:
            msg              = MIMEMultipart("alternative")
            msg["From"]      = self._smtp_user
            msg["To"]        = to
            msg["Subject"]   = subject
            msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._smtp_user, self._smtp_pass)
                server.send_message(msg)
            logger.info("Email sent to %s", to)
            return True
        except Exception as exc:
            logger.error("Email failed to %s: %s", to, exc)
            return False

    # ------------------------------------------------------------------
    # High-level notifiers
    # ------------------------------------------------------------------

    def notify_accident(
        self,
        incident_id:   int,
        incident_type: str,
        camera_id:     str,
        timestamp:     str,
        plate:         str,
        dashboard_url: str,
        owner_phone:   Optional[str] = None,
        owner_email:   Optional[str] = None,
        owner_name:    Optional[str] = None,
    ) -> list[dict]:
        """
        Send SMS + email to emergency contacts (always) and to the vehicle
        owner (if their details are provided).

        Returns list of alert dicts: {sent_to, channel, status, message_preview}
        """
        logs: list[dict] = []

        sms_msg = (
            f"🚨 RoadGuard AI Alert\n"
            f"Incident: {incident_type.replace('_',' ').title()}\n"
            f"Camera: {camera_id}  |  Plate: {plate or 'unknown'}\n"
            f"Time: {timestamp}\n"
            f"Dashboard: {dashboard_url}/incidents/{incident_id}"
        )

        email_body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#222;">
        <h2 style="color:#c0392b;">🚨 RoadGuard AI — {incident_type.replace('_',' ').title()}</h2>
        <table>
          <tr><td><b>Camera:</b></td><td>{camera_id}</td></tr>
          <tr><td><b>Plate:</b></td><td>{plate or 'Unknown'}</td></tr>
          <tr><td><b>Time:</b></td><td>{timestamp}</td></tr>
          <tr><td><b>Status:</b></td><td>Pending investigation</td></tr>
        </table>
        <br/>
        <a href="{dashboard_url}/incidents/{incident_id}"
           style="background:#c0392b;color:#fff;padding:8px 18px;border-radius:4px;text-decoration:none;">
          View Incident
        </a>
        </body></html>
        """

        # Emergency contacts
        for phone in self._emergency_phones:
            ok  = self.send_sms(phone, sms_msg)
            logs.append({"sent_to": phone, "channel": "sms",
                         "status": "sent" if ok else "failed",
                         "message_preview": sms_msg[:120]})

        for email in self._emergency_emails:
            ok  = self.send_email(email, f"RoadGuard Alert – {incident_type}", email_body)
            logs.append({"sent_to": email, "channel": "email",
                         "status": "sent" if ok else "failed",
                         "message_preview": sms_msg[:120]})

        # Vehicle owner
        if owner_phone:
            owner_sms = (
                f"RoadGuard AI: Your vehicle ({plate}) was involved in a "
                f"{incident_type.replace('_',' ')} at {timestamp}. "
                f"Ref: {dashboard_url}/incidents/{incident_id}"
            )
            ok = self.send_sms(owner_phone, owner_sms)
            logs.append({"sent_to": owner_phone, "channel": "sms",
                         "status": "sent" if ok else "failed",
                         "message_preview": owner_sms[:120]})

        if owner_email:
            ok = self.send_email(
                owner_email,
                f"RoadGuard – Your vehicle incident #{incident_id}",
                email_body.replace(
                    "<h2", f"<p>Dear {owner_name or 'Vehicle Owner'},</p><h2"
                ),
            )
            logs.append({"sent_to": owner_email, "channel": "email",
                         "status": "sent" if ok else "failed",
                         "message_preview": sms_msg[:120]})

        return logs