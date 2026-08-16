"""
Bespoke Dark Theme & Medical Audio Console Design System for VocalStrong AI.
Crafted for clinical speech pathology biofeedback with high contrast,
refined typography, teleprompter prompter views, and precision data visualization.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

/* Global Reset & Base Typography */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #080D1A !important;
    color: #E2E8F0 !important;
}

/* Sidebar Dark Styling */
section[data-testid="stSidebar"] {
    background: #0B1222 !important;
    border-right: 1px solid #1E293B !important;
}

section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F8FAFC !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.01em !important;
}

/* Header Audio Console Banner */
.console-header {
    background: linear-gradient(135deg, #0F172A 0%, #111C33 50%, #0A1124 100%);
    border: 1px solid #1E2E4A;
    border-radius: 16px;
    padding: 1.6rem 2.0rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.console-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D4B2, #38BDF8, #6366F1);
}

.console-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 178, 0.12);
    border: 1px solid rgba(0, 212, 178, 0.3);
    color: #00D4B2;
    padding: 0.25rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.console-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}

.console-subtitle {
    font-size: 0.98rem;
    color: #94A3B8;
    margin: 0;
    line-height: 1.5;
    font-weight: 400;
}

/* ========================================================================= */
/* CENTER TELEPROMPTER & READING STAGE (HERO PROMPTER)                       */
/* ========================================================================= */
.teleprompter-card {
    background: linear-gradient(180deg, #0F1A30 0%, #0B1325 100%);
    border: 2px solid #22375D;
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.45);
    position: relative;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.teleprompter-card.active-recording {
    border-color: #00D4B2;
    box-shadow: 0 0 35px rgba(0, 212, 178, 0.25), 0 10px 35px rgba(0, 0, 0, 0.5);
    background: linear-gradient(180deg, #0C1E38 0%, #08152B 100%);
}

.teleprompter-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.0rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1E2E4A;
}

.teleprompter-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 8px;
}

.target-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 178, 0.15);
    border: 1px solid rgba(0, 212, 178, 0.35);
    color: #00D4B2;
    padding: 0.3rem 0.9rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
}

.prompter-text-body {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.65;
    letter-spacing: -0.01em;
    padding: 1.2rem 1.4rem;
    background: rgba(8, 15, 29, 0.85);
    border: 1px solid #1A2845;
    border-radius: 12px;
    margin: 0.8rem 0;
    min-height: 90px;
    display: flex;
    align-items: center;
}

.prompter-footer-tips {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 0.6rem;
}

/* Metric / Telemetry HUD Cards */
.hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.hud-card {
    background: #0F172A;
    border: 1px solid #1E2E4A;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    position: relative;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.hud-card:hover {
    border-color: #2E456E;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
}

.hud-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.4rem;
}

.hud-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    line-height: 1.1;
}

.hud-unit {
    font-size: 1.0rem;
    font-weight: 500;
    color: #64748B;
    margin-left: 2px;
}

.hud-footer {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* Panel Containers */
.workstation-panel {
    background: #0F172A;
    border: 1px solid #1E2E4A;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1E293B;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Clinical Recommendation Cards */
.clinical-report-card {
    background: #111C33;
    border: 1px solid #1E2E4A;
    border-left-width: 5px;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
}

.clinical-report-card.critical {
    border-left-color: #F43F5E;
    background: linear-gradient(90deg, rgba(244, 63, 94, 0.08) 0%, #111C33 40%);
}

.clinical-report-card.warning {
    border-left-color: #F59E0B;
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, #111C33 40%);
}

.clinical-report-card.optimal {
    border-left-color: #10B981;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, #111C33 40%);
}

.clinical-report-card.info {
    border-left-color: #38BDF8;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.08) 0%, #111C33 40%);
}

.report-category {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.2rem;
}

.report-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.4rem;
}

.report-finding {
    font-size: 0.95rem;
    color: #CBD5E1;
    line-height: 1.5;
    margin-bottom: 0.8rem;
}

.report-rationale {
    font-size: 0.86rem;
    color: #94A3B8;
    margin-bottom: 0.8rem;
    padding-left: 0.8rem;
    border-left: 2px solid #334155;
    font-style: italic;
}

.directives-container {
    background: #0B1222;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1.0rem 1.2rem;
    margin-top: 0.6rem;
}

.directives-heading {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #00D4B2;
    margin-bottom: 0.5rem;
}

.directives-list {
    margin: 0;
    padding-left: 1.2rem;
}

.directives-list li {
    font-size: 0.92rem;
    color: #E2E8F0;
    margin-bottom: 0.45rem;
    line-height: 1.5;
}

/* Sarvam Transcription Box */
.sarvam-box {
    background: #0B1222;
    border: 1px solid #1E2E4A;
    border-radius: 12px;
    padding: 1.4rem;
}

.transcript-quote {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F8FAFC;
    line-height: 1.5;
    padding: 1.1rem;
    background: #111C33;
    border: 1px solid #1E293B;
    border-radius: 10px;
    margin: 0.8rem 0;
    font-style: italic;
}

/* Custom Styled Big Action Button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D4B2 0%, #0D9488 100%) !important;
    color: #041017 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.0rem !important;
    box-shadow: 0 4px 25px rgba(0, 212, 178, 0.4) !important;
    transition: all 0.25s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 35px rgba(0, 212, 178, 0.6) !important;
    color: #000000 !important;
}

div.stButton > button[kind="secondary"] {
    background: #131E35 !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E2E4A !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #38BDF8 !important;
    color: #38BDF8 !important;
    background: #172440 !important;
}

/* Custom Audio Player Dark Restyle */
audio {
    width: 100%;
    height: 40px;
    border-radius: 8px;
    filter: invert(90%) hue-rotate(180deg);
}

/* Download Buttons */
div.stDownloadButton > button {
    background: #131E35 !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E2E4A !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

div.stDownloadButton > button:hover {
    border-color: #00D4B2 !important;
    color: #00D4B2 !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #080D1A;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
"""
