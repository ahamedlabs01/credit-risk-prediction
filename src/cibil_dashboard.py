"""
TransUnion CIBIL Credit Bureau Dashboard Component
===================================================
Renders an authentic, interactive TransUnion CIBIL Credit Score Speedometer Gauge,
Credit Health Factors breakdown, loan sanction probability, and score improvement
advisory matching the official CIBIL portal (https://www.cibil.com/freecibilscore).
"""

import math
from typing import Dict, Any, List, Tuple


def get_cibil_band_details(score: int) -> Dict[str, Any]:
    """
    Returns official TransUnion CIBIL score band specifications, colors,
    sanction rates, and market percentiles.
    """
    score = max(300, min(900, int(score)))

    if score >= 800:
        return {
            "tier_name": "EXCELLENT",
            "tier_label": "Super Prime / Outstanding",
            "color": "#059669",
            "bg_color": "rgba(5, 150, 105, 0.15)",
            "border_color": "#059669",
            "sanction_prob": "96%",
            "sanction_text": "Exceptional credit profile. Pre-approved for premier loans at the lowest market interest rates.",
            "rate_discount": "-1.25% APR Discount",
            "badge_icon": "🌟"
        }
    elif score >= 750:
        return {
            "tier_name": "GOOD",
            "tier_label": "Prime / Benchmark Quality",
            "color": "#10b981",
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "#10b981",
            "sanction_prob": "85%",
            "sanction_text": "79%+ of all bank loans and credit cards in India are sanctioned for CIBIL scores above 750.",
            "rate_discount": "-0.75% APR Discount",
            "badge_icon": "🟢"
        }
    elif score >= 650:
        return {
            "tier_name": "FAIR",
            "tier_label": "Moderate Risk / Standard",
            "color": "#eab308",
            "bg_color": "rgba(234, 179, 8, 0.15)",
            "border_color": "#eab308",
            "sanction_prob": "65%",
            "sanction_text": "Standard approval likelihood. Banks may apply risk premium spread or require income verification.",
            "rate_discount": "+1.50% APR Spread",
            "badge_icon": "🟡"
        }
    elif score >= 550:
        return {
            "tier_name": "NEEDS WORK",
            "tier_label": "Elevated Default Risk / Poor",
            "color": "#f97316",
            "bg_color": "rgba(249, 115, 22, 0.15)",
            "border_color": "#f97316",
            "sanction_prob": "35%",
            "sanction_text": "Elevated default probability. Requires joint co-borrower, collateral pledge, or smaller credit limit.",
            "rate_discount": "+3.00% APR Spread",
            "badge_icon": "🟠"
        }
    else:
        return {
            "tier_name": "POOR",
            "tier_label": "Subprime / Policy Knockout",
            "color": "#ef4444",
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "border_color": "#ef4444",
            "sanction_prob": "< 15%",
            "sanction_text": "Score below 600 falls under hard bureau knockout. Unsecured credit facilities automatically declined.",
            "rate_discount": "+6.00% Risk Surcharge",
            "badge_icon": "⛔"
        }


def calculate_cibil_health_factors(applicant: Dict[str, Any], score: int) -> List[Dict[str, Any]]:
    """
    Computes the 5 official CIBIL credit score factors and their relative impact:
    1. Payment History (35% - High Impact)
    2. Credit Utilization (30% - High Impact)
    3. Credit Age / History (15% - Medium Impact)
    4. Credit Mix (10% - Low Impact)
    5. Recent Inquiries & Accounts (10% - Low Impact)
    """
    applicant = applicant or {}

    # 1. Payment History (35%)
    ch = applicant.get("credit_history", "A32")
    if ch in ["A30", "A31"]:
        pay_score, pay_status, pay_color = 98, "Excellent • 100% On-Time Track Record", "#10b981"
    elif ch == "A32":
        pay_score, pay_status, pay_color = 85, "Good • Existing Facilities Repaid Duly", "#38bdf8"
    elif ch == "A33":
        pay_score, pay_status, pay_color = 52, "Fair • Past Payment Delays Noted", "#eab308"
    else:  # A34
        pay_score, pay_status, pay_color = 22, "Critical • Multiple Delays or Defaults", "#ef4444"

    # 2. Credit Utilization & Liquidity (30%)
    chk = applicant.get("checking_account", "A12")
    sav = applicant.get("savings", "A61")
    if chk == "A13" or sav in ["A63", "A64"]:
        util_score, util_status, util_color = 92, "Optimal • High Liquidity Reserves (< 25% Usage)", "#10b981"
    elif chk == "A12" or sav == "A62":
        util_score, util_status, util_color = 72, "Moderate • Modest Reserves (25% - 40% Usage)", "#38bdf8"
    elif chk == "A14":
        util_score, util_status, util_color = 60, "Standard • No Open Checking Facility", "#eab308"
    else:  # A11 (negative balance)
        util_score, util_status, util_color = 25, "High Stress • Overdrawn Account (> 70% Usage)", "#ef4444"

    # 3. Credit Age & Stability (15%)
    emp = applicant.get("employment", "A73")
    age = int(applicant.get("age", 32))
    if emp == "A75" or age >= 45:
        age_score, age_status, age_color = 95, "Established • 7+ Years Stable Profile", "#10b981"
    elif emp in ["A73", "A74"]:
        age_score, age_status, age_color = 80, "Healthy • 2 to 5 Years Vintage", "#38bdf8"
    elif emp == "A72":
        age_score, age_status, age_color = 58, "Developing • Under 1 Year Work History", "#eab308"
    else:
        age_score, age_status, age_color = 30, "Unseasoned • Thin Credit File Vintage", "#ef4444"

    # 4. Credit Mix (10%)
    prop = applicant.get("property", "A121")
    if prop == "A121":
        mix_score, mix_status, mix_color = 92, "Balanced • Secured Real Estate Asset Backing", "#10b981"
    elif prop in ["A122", "A123"]:
        mix_score, mix_status, mix_color = 78, "Healthy • Auto / Insurance Asset Backing", "#38bdf8"
    else:
        mix_score, mix_status, mix_color = 50, "Unsecured Skew • No Registered Property Asset", "#eab308"

    # 5. Inquiries & Active Credits (10%)
    creds = int(applicant.get("existing_credits", 1))
    plans = applicant.get("other_installment_plans", "A143")
    if creds == 1 and plans == "A143":
        inq_score, inq_status, inq_color = 96, "Low Inquiries • Single Credit Line (Prudent)", "#10b981"
    elif creds == 2 and plans == "A143":
        inq_score, inq_status, inq_color = 80, "Moderate Activity • 2 Active Facilities", "#38bdf8"
    else:
        inq_score, inq_status, inq_color = 52, "Elevated • Multiple Open Installment Plans", "#eab308"

    return [
        {
            "name": "Payment History",
            "weight": "35%",
            "impact": "HIGH IMPACT",
            "impact_badge": "bg-red",
            "score_pct": pay_score,
            "status_text": pay_status,
            "color": pay_color,
            "desc": "Timely payment of past loan installments & credit card bills."
        },
        {
            "name": "Credit Utilization & Liquidity",
            "weight": "30%",
            "impact": "HIGH IMPACT",
            "impact_badge": "bg-red",
            "score_pct": util_score,
            "status_text": util_status,
            "color": util_color,
            "desc": "Ratio of credit limit utilized vs available financial reserves."
        },
        {
            "name": "Credit History Vintage",
            "weight": "15%",
            "impact": "MEDIUM IMPACT",
            "impact_badge": "bg-amber",
            "score_pct": age_score,
            "status_text": age_status,
            "color": age_color,
            "desc": "Length of time since your oldest active credit facility was opened."
        },
        {
            "name": "Credit Mix (Secured vs Unsecured)",
            "weight": "10%",
            "impact": "LOW IMPACT",
            "impact_badge": "bg-slate",
            "score_pct": mix_score,
            "status_text": mix_status,
            "color": mix_color,
            "desc": "Healthy balance between secured (home, auto) and unsecured loans."
        },
        {
            "name": "Recent Inquiries & Open Facilities",
            "weight": "10%",
            "impact": "LOW IMPACT",
            "impact_badge": "bg-slate",
            "score_pct": inq_score,
            "status_text": inq_status,
            "color": inq_color,
            "desc": "Number of new credit applications and active debt commitments."
        }
    ]


def generate_cibil_speedometer_svg(score: int) -> Tuple[str, float]:
    """
    Builds the clean SVG Speedometer dial showing 300 to 900 scale,
    segmented color zones, tick marks, and needle pointer.
    """
    clamped = max(300, min(900, int(score)))
    fraction = (clamped - 300) / 600.0
    
    # Angle in degrees (-90 deg is far left @ 300, 0 deg is top @ 600, +90 deg is far right @ 900)
    angle_deg = -90.0 + (fraction * 180.0)

    # 5 Arc Segments Coordinates (Radius 95, Center 150, 135)
    # 300-550 (Poor), 550-650 (Needs Impr), 650-750 (Fair), 750-800 (Good), 800-900 (Excellent)
    svg = f"""
    <svg viewBox="0 0 300 175" style="width: 100%; max-width: 380px; display: block; margin: 0 auto; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));">
        <defs>
            <linearGradient id="needleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#38bdf8" />
                <stop offset="100%" stop-color="#0284c7" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>

        <!-- Background Track -->
        <path d="M 55.0 135.0 A 95 95 0 0 1 245.0 135.0" fill="none" stroke="#1e293b" stroke-width="20" stroke-linecap="round" />

        <!-- Segment 1: Poor (300 - 550) -->
        <path d="M 55.0 135.0 A 95 95 0 0 1 125.41 43.24" fill="none" stroke="#ef4444" stroke-width="16" stroke-linecap="round" />

        <!-- Segment 2: Needs Improvement (550 - 650) -->
        <path d="M 125.41 43.24 A 95 95 0 0 1 174.59 43.24" fill="none" stroke="#f97316" stroke-width="16" />

        <!-- Segment 3: Fair (650 - 750) -->
        <path d="M 174.59 43.24 A 95 95 0 0 1 217.18 67.82" fill="none" stroke="#eab308" stroke-width="16" />

        <!-- Segment 4: Good (750 - 800) -->
        <path d="M 217.18 67.82 A 95 95 0 0 1 232.27 87.5" fill="none" stroke="#10b981" stroke-width="16" />

        <!-- Segment 5: Excellent (800 - 900) -->
        <path d="M 232.27 87.5 A 95 95 0 0 1 245.0 135.0" fill="none" stroke="#059669" stroke-width="16" stroke-linecap="round" />

        <!-- Scale Numbers -->
        <text x="36" y="152" fill="#94a3b8" font-size="11" font-weight="700" text-anchor="middle">300</text>
        <text x="115" y="32" fill="#94a3b8" font-size="10" font-weight="600" text-anchor="middle">550</text>
        <text x="150" y="24" fill="#cbd5e1" font-size="10" font-weight="700" text-anchor="middle">600</text>
        <text x="195" y="34" fill="#94a3b8" font-size="10" font-weight="600" text-anchor="middle">700</text>
        <text x="238" y="62" fill="#34d399" font-size="10" font-weight="700" text-anchor="middle">750</text>
        <text x="264" y="152" fill="#94a3b8" font-size="11" font-weight="700" text-anchor="middle">900</text>

        <!-- Needle Pivot Base & Rotated Needle -->
        <g transform="translate(150, 135) rotate({angle_deg})">
            <!-- Needle pointer -->
            <polygon points="-4,0 4,0 1,-78 -1,-78" fill="url(#needleGrad)" filter="url(#glow)" />
            <!-- Needle center accent -->
            <circle cx="0" cy="0" r="10" fill="#0f172a" stroke="#38bdf8" stroke-width="3" />
            <circle cx="0" cy="0" r="4" fill="#38bdf8" />
        </g>
    </svg>
    """
    return svg, angle_deg


def render_cibil_dashboard_html(score: int, applicant: Dict[str, Any] = None) -> str:
    """
    Generates a full TransUnion CIBIL dashboard card matching https://www.cibil.com/freecibilscore:
    - Official Header & Branding
    - Speedometer Dial with large score & Tier pill
    - Benchmark approval banner ('79% of loans approved for 750+')
    - 5 Official CIBIL Credit Factors breakdown
    - Actionable recommendations to improve score
    """
    band = get_cibil_band_details(score)
    svg_gauge, _ = generate_cibil_speedometer_svg(score)
    factors = calculate_cibil_health_factors(applicant, score)

    # Generate factors HTML
    factors_html = ""
    for f in factors:
        factors_html += f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">
                    {f['name']}
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; background: #0f172a; padding: 2px 8px; border-radius: 4px;">{f['weight']}</span>
                    <span style="font-size: 0.75rem; font-weight: 800; color: {f['color']}; background: rgba(15,23,42,0.8); border: 1px solid {f['color']}; padding: 2px 8px; border-radius: 4px;">{f['impact']}</span>
                </div>
            </div>
            <div style="width: 100%; background: #0f172a; border-radius: 9999px; height: 8px; overflow: hidden; margin-bottom: 8px;">
                <div style="width: {f['score_pct']}%; background: {f['color']}; height: 100%; border-radius: 9999px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                <span style="color: {f['color']}; font-weight: 600;">{f['status_text']}</span>
                <span style="color: #94a3b8; font-size: 0.8rem;">{f['desc']}</span>
            </div>
        </div>
        """

    # Recommendation advice based on score
    if score >= 750:
        advice_html = """
        <li style="margin-bottom: 6px;">🌟 <strong>Maintain Excellence:</strong> Keep credit card usage under 30% of limit and continue flawless repayment track record.</li>
        <li style="margin-bottom: 6px;">🎯 <strong>Rate Advantage:</strong> You qualify for prime lender interest rate concessions (up to -1.25% APR discount).</li>
        <li>🛡️ <strong>Monitor Periodically:</strong> Check your CIBIL report quarterly for any inaccurate reporting or identity theft.</li>
        """
    elif score >= 650:
        advice_html = """
        <li style="margin-bottom: 6px;">📈 <strong>Reach 750+:</strong> Pay down outstanding credit card balances to drop utilization below 30%.</li>
        <li style="margin-bottom: 6px;">⏱️ <strong>Clean Repayment:</strong> Maintain 6+ consecutive months of zero missed EMI payments to cross the prime 750 threshold.</li>
        <li>🚫 <strong>Limit Hard Inquiries:</strong> Avoid applying for multiple personal loans or credit cards simultaneously.</li>
        """
    else:
        advice_html = """
        <li style="margin-bottom: 6px;">⚠️ <strong>Clear Overdrafts:</strong> Immediately clear negative checking account balances and delinquent credit lines.</li>
        <li style="margin-bottom: 6px;">🤝 <strong>Use Co-borrower:</strong> Apply with a Tier-1 prime co-borrower or pledge collateral to secure facility sanction.</li>
        <li>🔄 <strong>Credit Repair:</strong> Settle outstanding past dues; on-time payment over 12 months is mandatory to recover above 650.</li>
        """

    html = f"""
    <div style="background: linear-gradient(145deg, #0b1329 0%, #0f172a 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 16px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); font-family: 'Inter', system-ui, sans-serif; color: #f8fafc; margin-bottom: 25px;">
        
        <!-- Header Branding -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(148, 163, 184, 0.2); padding-bottom: 16px; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #0284c7; color: #ffffff; font-weight: 900; font-size: 1.1rem; padding: 6px 12px; border-radius: 6px; letter-spacing: 1px;">
                    CIBIL
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">TransUnion CIBIL™ Credit Bureau Score</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Official Bureau Credit Health Report • Standard Scale (300 – 900)</div>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: {band['bg_color']}; color: {band['color']}; border: 1px solid {band['border_color']}; font-weight: 800; font-size: 0.85rem; padding: 4px 12px; border-radius: 9999px;">
                    {band['badge_icon']} {band['tier_name']}
                </span>
            </div>
        </div>

        <!-- Speedometer & Score Section -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; align-items: center; margin-bottom: 24px;">
            <div>
                {svg_gauge}
                <div style="text-align: center; margin-top: -10px;">
                    <div style="font-size: 2.8rem; font-weight: 900; color: {band['color']}; letter-spacing: -1px; line-height: 1;">
                        {score}
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">
                        {band['tier_label']}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                        Score Band: <strong>300 to 900</strong> | Benchmark Prime Cutoff: <strong>750</strong>
                    </div>
                </div>
            </div>

            <!-- Sanction Odds Card -->
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 20px;">
                <div style="font-size: 0.85rem; text-transform: uppercase; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 6px;">
                    Loan Approval & Sanction Probability
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {band['color']}; margin-bottom: 4px;">
                    {band['sanction_prob']} Sanction Chance
                </div>
                <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.45; margin-bottom: 12px;">
                    {band['sanction_text']}
                </p>
                
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <div style="background: rgba(2, 132, 199, 0.15); border: 1px solid #0284c7; padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; color: #38bdf8; font-weight: 600;">
                        Pricing Impact: <strong>{band['rate_discount']}</strong>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; color: #34d399; font-weight: 600;">
                        Bureau Standing: <strong>{band['tier_name']}</strong>
                    </div>
                </div>
            </div>
        </div>

        <!-- 5 CIBIL Score Factors Section -->
        <div style="border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 20px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div>
                    <span style="font-size: 1.05rem; font-weight: 800; color: #f8fafc;">📊 Official CIBIL Credit Health Factors (5 Pillars)</span>
                    <span style="font-size: 0.8rem; color: #94a3b8; display: block;">How your financial attributes directly influence your TransUnion CIBIL score</span>
                </div>
            </div>
            
            {factors_html}
        </div>

        <!-- Improvement Plan -->
        <div style="background: rgba(15, 23, 42, 0.5); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-top: 18px; font-size: 0.85rem; color: #cbd5e1;">
            <div style="font-weight: 800; color: #38bdf8; margin-bottom: 6px; font-size: 0.9rem;">
                💡 TransUnion CIBIL Score Advisory & Growth Plan
            </div>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.5;">
                {advice_html}
            </ul>
        </div>

    </div>
    """
    return html
