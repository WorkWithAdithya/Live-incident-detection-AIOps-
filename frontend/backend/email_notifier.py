"""
frontend/backend/email_notifier.py
------------------------------------
Sends email alerts to the admin when system metrics breach
WARNING or CRITICAL thresholds set by the user.

Uses Python's built-in smtplib — no extra dependencies.
Supports Gmail (with App Password), Outlook, or any SMTP server.

Config via environment variables in log_generator/.env:

    ALERT_EMAIL_FROM      = your-sender@gmail.com
    ALERT_EMAIL_PASSWORD  = your-app-password
    ALERT_EMAIL_TO        = admin@yourcompany.com
    ALERT_SMTP_HOST       = smtp.gmail.com          (default)
    ALERT_SMTP_PORT       = 587                     (default)

    # Cooldown: min seconds between emails for the same severity
    # Prevents flooding if metrics stay high for a long time
    ALERT_COOLDOWN_SECONDS = 300                    (default 5 min)

Gmail setup:
    1. Enable 2FA on your Google account
    2. Go to Google Account → Security → App Passwords
    3. Create an App Password for "Mail"
    4. Use that 16-char password as ALERT_EMAIL_PASSWORD
"""

import os
import smtplib
import threading
from email.mime.text    import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime           import datetime
from dotenv             import load_dotenv
from pathlib            import Path

# Resolve .env
_ENV_PATH = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "log_generator" / ".env"
)
load_dotenv(dotenv_path=_ENV_PATH)

# ── Config ────────────────────────────────────────────────────────────────────
# Strips inline comments from .env values (e.g. "value  # comment")
def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, default).strip()
    for sep in ("  #", " #", "\t#"):
        if sep in val:
            val = val[:val.index(sep)].strip()
    return val

EMAIL_FROM        = _env("ALERT_EMAIL_FROM")
EMAIL_PASSWORD    = _env("ALERT_EMAIL_PASSWORD")
EMAIL_TO          = _env("ALERT_EMAIL_TO")
SMTP_HOST         = _env("ALERT_SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT         = int(_env("ALERT_SMTP_PORT") or "587")
COOLDOWN_SECONDS  = int(_env("ALERT_COOLDOWN_SECONDS") or "300")


class EmailNotifier:
    """
    Thread-safe email notifier with per-severity cooldown.
    Cooldown prevents repeated emails if metrics stay breached.
    """

    def __init__(self):
        self._lock          = threading.Lock()
        self._last_sent: dict[str, datetime] = {}   # severity → last sent time
        self._enabled       = self._check_config()

    def _check_config(self) -> bool:
        # Print loaded values for easy debugging (password masked)
        print(f"   ALERT_EMAIL_FROM     : {EMAIL_FROM or '(not set)'}")
        print(f"   ALERT_EMAIL_PASSWORD : {'*' * len(EMAIL_PASSWORD) if EMAIL_PASSWORD else '(not set)'}")
        print(f"   ALERT_EMAIL_TO       : {EMAIL_TO or '(not set)'}")
        print(f"   ALERT_SMTP_HOST      : {SMTP_HOST}")
        print(f"   ALERT_SMTP_PORT      : {SMTP_PORT}")
        print(f"   ALERT_COOLDOWN_SECS  : {COOLDOWN_SECONDS}")

        if not EMAIL_FROM or not EMAIL_PASSWORD or not EMAIL_TO:
            print(
                "⚠️  Email alerts disabled — set ALERT_EMAIL_FROM, "
                "ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO in log_generator/.env"
            )
            return False

        # Warn if password looks like a regular password (not a 16-char app password)
        clean_pw = EMAIL_PASSWORD.replace(" ", "")
        if len(clean_pw) != 16:
            print(
                f"⚠️  ALERT_EMAIL_PASSWORD is {len(clean_pw)} chars "
                f"(expected 16 for Gmail App Password). "
                f"Regular Gmail passwords are blocked by Google."
            )

        print(
            f"✅ Email alerts enabled → {EMAIL_TO}  "
            f"(cooldown: {COOLDOWN_SECONDS}s)"
        )
        return True

    def _is_on_cooldown(self, severity: str) -> bool:
        last = self._last_sent.get(severity)
        if last is None:
            return False
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed < COOLDOWN_SECONDS

    def _build_email(
        self,
        severity:   str,
        metrics:    dict,
        exceeded:   list[str],
        threshold_info: dict,
    ) -> MIMEMultipart:
        """Builds a clean plain-text + HTML email."""

        ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icon      = "⚠️" if severity == "WARNING" else "🔴"
        color_hex = "#facc15" if severity == "WARNING" else "#f87171"
        subject   = f"{icon} AIOps {severity}: System metric threshold breached"

        # ── Plain text ────────────────────────────────────────────────────────
        plain = f"""
AIOps Incident Detection — {severity} Alert
============================================
Time      : {ts}
Severity  : {severity}

Current Metric Values
---------------------
  CPU Usage    : {metrics.get('cpu', '—'):.2f}%
  Memory Usage : {metrics.get('memory', '—'):.2f}%
  Disk Usage   : {metrics.get('disk', '—'):.2f}%

Thresholds Breached
-------------------
{chr(10).join(f'  • {e}' for e in exceeded)}

Configured Limits
-----------------
{chr(10).join(f'  {k}: {v}%' for k, v in threshold_info.items() if v is not None)}

Action Required
---------------
Please investigate the system immediately.
This alert was generated by the AIOps LSTM Autoencoder Incident Detection System.

-- AIOps Dashboard
"""

        # ── HTML ──────────────────────────────────────────────────────────────
        exceeded_rows = "".join(
            f"<tr><td style='padding:4px 12px;color:#f87171;'>⚠ {e}</td></tr>"
            for e in exceeded
        )
        limit_rows = "".join(
            f"<tr><td style='padding:2px 12px;color:#888;font-size:12px;'>{k}</td>"
            f"<td style='padding:2px 12px;color:#e8e8e8;font-size:12px;'>{v}%</td></tr>"
            for k, v in threshold_info.items() if v is not None
        )

        html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0e0e0e;font-family:'IBM Plex Mono',monospace;color:#e8e8e8;">
  <div style="max-width:560px;margin:32px auto;background:#161616;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden;">

    <!-- Header -->
    <div style="background:{color_hex}15;border-bottom:2px solid {color_hex};padding:20px 24px;">
      <div style="font-size:11px;letter-spacing:0.1em;color:#888;margin-bottom:4px;">AIOPS INCIDENT DETECTION</div>
      <div style="font-size:22px;font-weight:600;color:{color_hex};">{icon} {severity} ALERT</div>
      <div style="font-size:11px;color:#888;margin-top:4px;">{ts}</div>
    </div>

    <!-- Metrics -->
    <div style="padding:20px 24px;border-bottom:1px solid #2a2a2a;">
      <div style="font-size:10px;letter-spacing:0.08em;color:#555;margin-bottom:12px;">CURRENT METRIC VALUES</div>
      <table style="width:100%;border-collapse:collapse;">
        <tr>
          <td style="padding:6px 0;color:#a78bfa;font-size:13px;">CPU Usage</td>
          <td style="padding:6px 0;text-align:right;font-size:18px;font-weight:600;color:#a78bfa;">{metrics.get('cpu', 0):.2f}%</td>
        </tr>
        <tr>
          <td style="padding:6px 0;color:#38bdf8;font-size:13px;">Memory Usage</td>
          <td style="padding:6px 0;text-align:right;font-size:18px;font-weight:600;color:#38bdf8;">{metrics.get('memory', 0):.2f}%</td>
        </tr>
        <tr>
          <td style="padding:6px 0;color:#fb923c;font-size:13px;">Disk Usage</td>
          <td style="padding:6px 0;text-align:right;font-size:18px;font-weight:600;color:#fb923c;">{metrics.get('disk', 0):.2f}%</td>
        </tr>
      </table>
    </div>

    <!-- Breached thresholds -->
    <div style="padding:20px 24px;border-bottom:1px solid #2a2a2a;">
      <div style="font-size:10px;letter-spacing:0.08em;color:#555;margin-bottom:10px;">THRESHOLDS BREACHED</div>
      <table style="width:100%;border-collapse:collapse;">
        {exceeded_rows}
      </table>
    </div>

    <!-- Configured limits -->
    <div style="padding:16px 24px;border-bottom:1px solid #2a2a2a;">
      <div style="font-size:10px;letter-spacing:0.08em;color:#555;margin-bottom:8px;">CONFIGURED LIMITS</div>
      <table style="width:100%;border-collapse:collapse;">
        {limit_rows}
      </table>
    </div>

    <!-- Footer -->
    <div style="padding:14px 24px;background:#111;font-size:10px;color:#555;text-align:center;">
      AIOps LSTM Autoencoder · Incident Detection System
    </div>
  </div>
</body>
</html>
"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"AIOps Alert <{EMAIL_FROM}>"
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        return msg

    def _send_async(self, msg: MIMEMultipart, severity: str):
        """Sends email in a background thread — never blocks the SSE stream."""
        def _send():
            try:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(EMAIL_FROM, EMAIL_PASSWORD)
                    server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
                print(f"📧 Alert email sent → {EMAIL_TO}  [{severity}]")
            except smtplib.SMTPAuthenticationError:
                print("❌ Email auth failed — check ALERT_EMAIL_FROM / ALERT_EMAIL_PASSWORD")
            except smtplib.SMTPException as e:
                print(f"❌ SMTP error: {e}")
            except Exception as e:
                print(f"❌ Email send error: {e}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    def notify(
        self,
        severity:       str,
        metrics:        dict,
        exceeded:       list[str],
        threshold_info: dict,
    ):
        """
        Public method — call this from the SSE stream whenever
        a WARNING or CRITICAL is detected.

        Args:
            severity       : "WARNING" or "CRITICAL"
            metrics        : { cpu, memory, disk } current values
            exceeded       : list of human-readable breach descriptions
            threshold_info : { 'CPU Warning': 70, 'CPU Critical': 85, ... }
        """
        if not self._enabled:
            return
        if severity not in ("WARNING", "CRITICAL"):
            return

        with self._lock:
            if self._is_on_cooldown(severity):
                remaining = COOLDOWN_SECONDS - (
                    datetime.now() - self._last_sent[severity]
                ).total_seconds()
                print(
                    f"📧 Email cooldown active for {severity} "
                    f"— {remaining:.0f}s remaining"
                )
                return

            self._last_sent[severity] = datetime.now()

        msg = self._build_email(severity, metrics, exceeded, threshold_info)
        self._send_async(msg, severity)

    def _build_forecast_email(
        self,
        severity      : str,
        breaches      : list,
        current_metrics: dict,
    ) -> MIMEMultipart:
        """
        Builds a forecast-specific email showing predicted future breaches.
        Different from _build_email which shows current breaches.
        """
        ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icon      = "⚠️" if severity == "WARNING" else "🔴"
        color_hex = "#facc15" if severity == "WARNING" else "#f87171"
        subject   = f"{icon} AIOps FORECAST: {severity} predicted — threshold breach incoming"

        # ── Plain text ────────────────────────────────────────────────────────
        breach_lines = "\n".join(
            f"  • {b['label']}: predicted {b['predicted_value']:.1f}% "
            f"(limit: {b['limit']}%) in ~{b['seconds_ahead']}s "
            f"at {b.get('predicted_at', '—')}"
            for b in breaches
        )

        plain = f"""
AIOps Incident Detection — FORECAST {severity} Alert
=====================================================
Time       : {ts}
Severity   : {severity}
Type       : PREDICTED (LSTM Forecaster — not yet occurred)

Current Metric Values
---------------------
  CPU Usage    : {current_metrics.get('cpu', 0):.2f}%
  Memory Usage : {current_metrics.get('memory', 0):.2f}%
  Disk Usage   : {current_metrics.get('disk', 0):.2f}%

Predicted Threshold Breaches
-----------------------------
{breach_lines}

Action Required
---------------
These are PREDICTED breaches based on the LSTM Forecaster model.
The system has not yet breached these thresholds — take preventive action now.

-- AIOps Dashboard (LSTM Forecaster)
"""

        # ── HTML breach rows ──────────────────────────────────────────────────
        breach_html = ""
        for b in breaches:
            sev_color = "#f87171" if b["severity"] == "CRITICAL" else "#facc15"
            breach_html += f"""
            <tr>
              <td style='padding:8px 0;color:{sev_color};font-weight:600;'>{b["label"]}</td>
              <td style='padding:8px 0;text-align:center;color:#e8e8e8;'>{b["predicted_value"]:.1f}%</td>
              <td style='padding:8px 0;text-align:center;color:{sev_color};'>&gt; {b["limit"]}%</td>
              <td style='padding:8px 0;text-align:right;color:#e8e8e8;'>~{b["seconds_ahead"]}s</td>
            </tr>
            <tr>
              <td colspan='4' style='padding:0 0 8px 0;font-size:11px;color:#888;'>
                Breach at: {b.get("predicted_at", "—")}
              </td>
            </tr>"""

        html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0e0e0e;font-family:'IBM Plex Mono',monospace;color:#e8e8e8;">
  <div style="max-width:560px;margin:32px auto;background:#161616;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden;">

    <!-- Header -->
    <div style="background:{color_hex}15;border-bottom:2px solid {color_hex};padding:20px 24px;">
      <div style="font-size:11px;letter-spacing:0.1em;color:#888;margin-bottom:4px;">AIOPS — LSTM FORECASTER ALERT</div>
      <div style="font-size:20px;font-weight:600;color:{color_hex};">{icon} {severity} BREACH PREDICTED</div>
      <div style="font-size:11px;color:#888;margin-top:4px;">{ts}</div>
    </div>

    <!-- Forecast notice -->
    <div style="padding:14px 24px;border-bottom:1px solid #2a2a2a;background:#1a2535;">
      <div style="font-size:11px;color:#38bdf8;">
        ℹ This is a PREDICTIVE alert — the LSTM model forecasts these values
        will breach your set thresholds. Take action before it happens.
      </div>
    </div>

    <!-- Current metrics -->
    <div style="padding:16px 24px;border-bottom:1px solid #2a2a2a;">
      <div style="font-size:10px;letter-spacing:0.08em;color:#555;margin-bottom:10px;">CURRENT VALUES (NOW)</div>
      <table style="width:100%;border-collapse:collapse;">
        <tr>
          <td style="padding:4px 0;color:#a78bfa;">CPU</td>
          <td style="padding:4px 0;text-align:right;font-weight:600;color:#a78bfa;">{current_metrics.get("cpu",0):.2f}%</td>
          <td style="padding:4px 0;color:#38bdf8;padding-left:20px;">Memory</td>
          <td style="padding:4px 0;text-align:right;font-weight:600;color:#38bdf8;">{current_metrics.get("memory",0):.2f}%</td>
          <td style="padding:4px 0;color:#fb923c;padding-left:20px;">Disk</td>
          <td style="padding:4px 0;text-align:right;font-weight:600;color:#fb923c;">{current_metrics.get("disk",0):.2f}%</td>
        </tr>
      </table>
    </div>

    <!-- Predicted breaches -->
    <div style="padding:20px 24px;border-bottom:1px solid #2a2a2a;">
      <div style="font-size:10px;letter-spacing:0.08em;color:#555;margin-bottom:10px;">PREDICTED BREACHES</div>
      <table style="width:100%;border-collapse:collapse;">
        <tr style="font-size:10px;color:#555;border-bottom:1px solid #2a2a2a;">
          <th style="text-align:left;padding-bottom:6px;">Metric</th>
          <th style="text-align:center;padding-bottom:6px;">Predicted</th>
          <th style="text-align:center;padding-bottom:6px;">Limit</th>
          <th style="text-align:right;padding-bottom:6px;">ETA</th>
        </tr>
        {breach_html}
      </table>
    </div>

    <!-- Footer -->
    <div style="padding:14px 24px;background:#111;font-size:10px;color:#555;text-align:center;">
      AIOps LSTM Forecaster · Predictive Incident Detection
    </div>
  </div>
</body>
</html>
"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"AIOps Forecast <{EMAIL_FROM}>"
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        return msg

    def notify_forecast_breach(
        self,
        severity        : str,
        breaches        : list,
        current_metrics : dict,
    ):
        """
        Sends a PREDICTIVE alert email when the LSTM Forecaster
        predicts a future threshold breach.

        Args:
            severity        : "WARNING" or "CRITICAL" (worst predicted)
            breaches        : list of forecast_breaches from inference_engine
                              [{ metric, label, severity, predicted_value,
                                 limit, seconds_ahead, predicted_at, criteria }]
            current_metrics : { cpu, memory, disk } — current values right now
        """
        if not self._enabled:
            return
        if severity not in ("WARNING", "CRITICAL"):
            return
        if not breaches:
            return

        # Use a separate cooldown key for forecast alerts
        cooldown_key = f"FORECAST_{severity}"

        with self._lock:
            if self._is_on_cooldown(cooldown_key):
                remaining = COOLDOWN_SECONDS - (
                    datetime.now() - self._last_sent[cooldown_key]
                ).total_seconds()
                print(
                    f"📧 Forecast email cooldown for {severity} "
                    f"— {remaining:.0f}s remaining"
                )
                return
            self._last_sent[cooldown_key] = datetime.now()

        msg = self._build_forecast_email(severity, breaches, current_metrics)
        self._send_async(msg, f"FORECAST_{severity}")
        print(
            f"📧 Forecast alert email queued → {EMAIL_TO}  "
            f"[{severity}] {len(breaches)} predicted breach(es)"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset_cooldown(self):
        """Call this if you want to force the next alert to send immediately."""
        with self._lock:
            self._last_sent.clear()


# Singleton — import and use this instance throughout the backend
notifier = EmailNotifier()