"""
Underwriting & Loan Terms Recommendation Engine
================================================
Provides risk-adjusted interest rate (APR) pricing, maximum credit limit sizing,
loan amortization schedules, risk covenant determinations, and executive Credit
Assessment Memo generation for credit risk prediction.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from typing import Dict, Any, List, Optional


@dataclass
class UnderwritingTerms:
    decision: str  # "Approved", "Conditionally Approved", "Declined"
    risk_grade: str  # "AAA", "AA", "A", "BBB", "BB", "B", "CCC"
    risk_tier: str  # "Tier 1 (Prime)", "Tier 2 (Moderate)", "Tier 3 (High)"
    recommended_apr: float  # e.g., 7.25 (%)
    max_safe_credit_limit: float  # e.g., 5000.0 (DM)
    requested_amount: float
    requested_duration: int
    monthly_payment_requested: float
    total_interest_requested: float
    total_cost_requested: float
    monthly_payment_recommended: float
    total_interest_recommended: float
    total_cost_recommended: float
    debt_service_burden: str  # e.g., "Manageable (12% of est. income)"
    conditions: List[str] = field(default_factory=list)
    docket_id: str = field(default_factory=lambda: f"CR-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}")
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))


def calculate_amortization(principal: float, annual_rate_pct: float, tenure_months: int) -> Dict[str, float]:
    """Calculate fixed monthly repayment, total interest, and total repayment amount."""
    if tenure_months <= 0 or principal <= 0:
        return {"monthly_payment": 0.0, "total_interest": 0.0, "total_payment": 0.0}
    
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate == 0:
        monthly_payment = principal / tenure_months
        total_payment = principal
        total_interest = 0.0
    else:
        # Standard annuity formula
        factor = (1.0 + monthly_rate) ** tenure_months
        monthly_payment = principal * (monthly_rate * factor) / (factor - 1.0)
        total_payment = monthly_payment * tenure_months
        total_interest = total_payment - principal

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2)
    }


def evaluate_underwriting_terms(
    applicant: Dict[str, Any],
    bad_probability: float,
    threshold: float = 0.30
) -> UnderwritingTerms:
    """
    Evaluates applicant profile and generates risk-based loan pricing,
    max credit sizing, and underwriting covenants.
    """
    requested_amount = float(applicant.get("credit_amount", 3000))
    requested_duration = int(applicant.get("duration", 24))
    installment_rate = int(applicant.get("installment_rate", 2))
    
    # -------------------------------------------------------------------------
    # 1. Determine Decision & Risk Grade
    # -------------------------------------------------------------------------
    if bad_probability < threshold * 0.70:
        decision = "Approved"
        if bad_probability < 0.12:
            risk_grade = "AAA"
            risk_tier = "Tier 1 (Prime)"
        elif bad_probability < 0.20:
            risk_grade = "AA"
            risk_tier = "Tier 1 (Prime)"
        else:
            risk_grade = "A"
            risk_tier = "Tier 1 (Prime - Standard)"
    elif bad_probability <= threshold * 1.15:
        decision = "Conditionally Approved"
        if bad_probability <= threshold:
            risk_grade = "BBB"
            risk_tier = "Tier 2 (Moderate Risk)"
        else:
            risk_grade = "BB"
            risk_tier = "Tier 2 (Elevated Risk)"
    else:
        decision = "Declined"
        if bad_probability < 0.60:
            risk_grade = "B"
            risk_tier = "Tier 3 (High Risk)"
        else:
            risk_grade = "CCC"
            risk_tier = "Tier 3 (Subprime / Critical Risk)"

    # -------------------------------------------------------------------------
    # 2. Risk-Adjusted Pricing (APR %)
    # -------------------------------------------------------------------------
    # Base rate for prime lending is 5.50%
    base_apr = 5.50
    # Risk spread scales with default probability
    risk_spread = bad_probability * 16.50
    
    # Specific adjustment factors
    modifiers = 0.0
    
    # Savings adjustment
    savings = applicant.get("savings", "A61")
    if savings == "A64":  # >= 1000 DM
        modifiers -= 0.75
    elif savings == "A63":  # 500-1000 DM
        modifiers -= 0.50
    elif savings in ["A61", "A65"]:  # < 100 DM or unknown
        modifiers += 0.50
        
    # Property collateral adjustment
    property_val = applicant.get("property", "A121")
    if property_val == "A121":  # Real estate
        modifiers -= 0.60
    elif property_val == "A124":  # No property
        modifiers += 0.85
        
    # Credit history adjustment
    credit_hist = applicant.get("credit_history", "A32")
    if credit_hist in ["A30", "A31"]:
        modifiers -= 0.35
    elif credit_hist == "A34":  # Critical account
        modifiers += 1.50
    elif credit_hist == "A33":  # Past delays
        modifiers += 0.90
        
    # Employment stability adjustment
    employment = applicant.get("employment", "A73")
    if employment == "A75":  # >= 7 years
        modifiers -= 0.50
    elif employment == "A71":  # Unemployed
        modifiers += 2.00
    elif employment == "A72":  # < 1 year
        modifiers += 0.75

    raw_apr = base_apr + risk_spread + modifiers
    # Clamp within realistic lending boundaries: 4.95% to 23.95%
    recommended_apr = round(max(4.95, min(raw_apr, 23.95)), 2)

    # -------------------------------------------------------------------------
    # 3. Maximum Safe Credit Limit Sizing (DM)
    # -------------------------------------------------------------------------
    # Base calculation uses collateral, savings tier, tenure, and risk score
    savings_mult = {
        "A61": 0.85, "A62": 1.00, "A63": 1.25, "A64": 1.60, "A65": 0.90
    }.get(savings, 1.0)
    
    property_mult = {
        "A121": 1.40, "A122": 1.15, "A123": 1.00, "A124": 0.70
    }.get(property_val, 1.0)
    
    emp_mult = {
        "A71": 0.40, "A72": 0.75, "A73": 1.00, "A74": 1.20, "A75": 1.35
    }.get(employment, 1.0)

    # Composite financial capacity coefficient
    capacity_factor = savings_mult * property_mult * emp_mult

    if decision == "Approved":
        # Can approve full requested amount or offer higher pre-approved ceiling
        max_safe_credit_limit = round(max(requested_amount, min(requested_amount * 1.30 * capacity_factor, 25000.0)), -1)
    elif decision == "Conditionally Approved":
        # Prudent scale back to prevent over-leverage
        safe_calc = requested_amount * 0.75 * min(capacity_factor, 1.0)
        max_safe_credit_limit = round(max(500.0, min(safe_calc, 12000.0)), -1)
    else:  # Declined
        # Maximum micro-credit / collateralized recovery limit only
        safe_calc = requested_amount * 0.35 * min(capacity_factor, 0.75)
        max_safe_credit_limit = round(max(250.0, min(safe_calc, 4000.0)), -1)

    # -------------------------------------------------------------------------
    # 4. Loan Amortization Calculations
    # -------------------------------------------------------------------------
    amort_req = calculate_amortization(requested_amount, recommended_apr, requested_duration)
    
    # Recommended scenario: if conditional/declined, evaluate recommended safe amount
    rec_amount_eval = requested_amount if decision == "Approved" else min(requested_amount, max_safe_credit_limit)
    rec_duration_eval = min(requested_duration, 36) if decision != "Approved" else requested_duration
    amort_rec = calculate_amortization(rec_amount_eval, recommended_apr, rec_duration_eval)

    # Debt service burden label
    if installment_rate >= 4:
        debt_service_burden = "Heavy (4% bracket - high disposable income commitment)"
    elif installment_rate == 3:
        debt_service_burden = "Moderate-High (3% bracket - standard risk exposure)"
    elif installment_rate == 2:
        debt_service_burden = "Balanced (2% bracket - healthy repayment headroom)"
    else:
        debt_service_burden = "Conservative (1% bracket - minimal income stress)"

    # -------------------------------------------------------------------------
    # 5. Underwriting Conditions & Covenants
    # -------------------------------------------------------------------------
    conditions = []
    
    if decision == "Approved":
        conditions.append("Standard identity, residential verification, and income validation.")
        if requested_amount > 10000:
            conditions.append("Senior underwriter countersignature required for credit facility exceeding 10,000 DM.")
        if savings in ["A61", "A65"]:
            conditions.append("Establish automated direct-debit on primary checking account for monthly installments.")
    elif decision == "Conditionally Approved":
        if requested_amount > max_safe_credit_limit:
            conditions.append(f"Cap facility amount at recommended limit of {max_safe_credit_limit:,.0f} DM (down from requested {requested_amount:,.0f} DM).")
        if requested_duration > 36:
            conditions.append("Shorten loan maturity to 24–36 months to reduce amortization default risk.")
        if applicant.get("other_debtors") == "A101":
            conditions.append("Require a creditworthy co-applicant or guarantor (Code A102/A103).")
        if property_val == "A124":
            conditions.append("Pledge secondary collateral or vehicle title to support the loan.")
        if applicant.get("checking_account") == "A11":
            conditions.append("Overdraft clearance and 60-day positive checking balance history required prior to disbursement.")
    else:  # Declined
        conditions.append("Primary application declined: Default risk exceeds institution risk appetite.")
        conditions.append(f"Consider restructuring request as fully collateralized loan capped at {max_safe_credit_limit:,.0f} DM.")
        conditions.append("Mandatory joint co-borrower with Tier 1 (Prime) standing required for appeal.")
        if applicant.get("credit_history") in ["A33", "A34"]:
            conditions.append("Applicant must provide documentation resolving outstanding liabilities at other financial institutions.")

    return UnderwritingTerms(
        decision=decision,
        risk_grade=risk_grade,
        risk_tier=risk_tier,
        recommended_apr=recommended_apr,
        max_safe_credit_limit=max_safe_credit_limit,
        requested_amount=requested_amount,
        requested_duration=requested_duration,
        monthly_payment_requested=amort_req["monthly_payment"],
        total_interest_requested=amort_req["total_interest"],
        total_cost_requested=amort_req["total_payment"],
        monthly_payment_recommended=amort_rec["monthly_payment"],
        total_interest_recommended=amort_rec["total_interest"],
        total_cost_recommended=amort_rec["total_payment"],
        debt_service_burden=debt_service_burden,
        conditions=conditions
    )


def generate_credit_memo_html(
    applicant: Dict[str, Any],
    labels_dict: Dict[str, str],
    terms: UnderwritingTerms,
    bad_probability: float,
    threshold: float
) -> str:
    """
    Generates a high-quality, professional Credit Assessment Memo in HTML
    with print-to-PDF optimizations and modern executive styling.
    """
    status_color = "#10b981" if terms.decision == "Approved" else ("#f59e0b" if terms.decision == "Conditionally Approved" else "#ef4444")
    status_bg = "rgba(16, 185, 129, 0.12)" if terms.decision == "Approved" else ("rgba(245, 158, 11, 0.12)" if terms.decision == "Conditionally Approved" else "rgba(239, 68, 68, 0.12)")

    conditions_li = "".join([f"<li>{c}</li>" for c in terms.conditions])

    # Format applicant profile rows
    def fmt(key: str) -> str:
        val = applicant.get(key, "N/A")
        if isinstance(val, str) and val in labels_dict:
            return f"{labels_dict[val]} <span style='color:#64748b; font-size:0.85em;'>({val})</span>"
        return str(val)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Credit Assessment Memorandum • {terms.docket_id}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    
    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
        padding: 32px;
        line-height: 1.5;
    }}

    .memo-container {{
        max-width: 860px;
        margin: 0 auto;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }}

    /* Header & Branding */
    .memo-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 2px solid #0f172a;
        padding-bottom: 24px;
        margin-bottom: 28px;
    }}

    .brand-title {{
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
    }}

    .brand-sub {{
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}

    .memo-meta {{
        text-align: right;
        font-size: 13px;
        color: #475569;
    }}

    .memo-meta strong {{
        color: #0f172a;
    }}

    /* Decision Banner */
    .decision-banner {{
        background: {status_bg};
        border-left: 5px solid {status_color};
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .decision-label {{
        font-size: 13px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .decision-value {{
        font-size: 22px;
        font-weight: 800;
        color: {status_color};
        margin-top: 2px;
    }}

    .risk-badge {{
        display: inline-block;
        background: #0f172a;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    /* Key Terms KPI Grid */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}

    .kpi-card {{
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}

    .kpi-label {{
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
    }}

    .kpi-sub {{
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }}

    /* Section Headings */
    .section-title {{
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 28px;
        margin-bottom: 16px;
    }}

    /* Comparison Table */
    .table-container {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 24px;
        font-size: 13px;
    }}

    .table-container th, .table-container td {{
        padding: 10px 14px;
        border: 1px solid #e2e8f0;
        text-align: left;
    }}

    .table-container th {{
        background-color: #f8fafc;
        font-weight: 700;
        color: #334155;
    }}

    .table-container tr:nth-child(even) td {{
        background-color: #fbfcfe;
    }}

    /* Conditions Checklist */
    .conditions-list {{
        list-style: none;
        padding: 0;
        margin-bottom: 30px;
    }}

    .conditions-list li {{
        position: relative;
        padding-left: 28px;
        margin-bottom: 10px;
        font-size: 13.5px;
        color: #334155;
    }}

    .conditions-list li::before {{
        content: "✓";
        position: absolute;
        left: 0;
        top: 0;
        font-weight: 800;
        color: {status_color};
    }}

    /* Sign-off Blocks */
    .signature-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 32px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px dashed #cbd5e1;
    }}

    .sign-line {{
        margin-top: 40px;
        border-bottom: 1px solid #0f172a;
    }}

    .sign-title {{
        font-size: 12px;
        font-weight: 600;
        color: #475569;
        margin-top: 6px;
    }}

    .disclaimer {{
        font-size: 11px;
        color: #94a3b8;
        margin-top: 30px;
        text-align: justify;
        line-height: 1.4;
    }}

    .print-actions {{
        margin-bottom: 20px;
        text-align: right;
    }}

    .btn-print {{
        background: #0f172a;
        color: #ffffff;
        border: none;
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
    }}

    .btn-print:hover {{
        background: #1e293b;
    }}

    @media print {{
        body {{
            background: #ffffff;
            padding: 0;
        }}
        .memo-container {{
            border: none;
            box-shadow: none;
            padding: 0;
            max-width: 100%;
        }}
        .print-actions {{
            display: none;
        }}
    }}
</style>
</head>
<body>

<div class="memo-container">
    <div class="print-actions">
        <button class="btn-print" onclick="window.print()">🖨️ Print Memo to PDF</button>
    </div>

    <div class="memo-header">
        <div>
            <div class="brand-title">🏦 CreditIQ Financial Systems</div>
            <div class="brand-sub">Credit Risk & Underwriting Committee Memorandum</div>
        </div>
        <div class="memo-meta">
            <div><strong>Docket No:</strong> {terms.docket_id}</div>
            <div><strong>Date:</strong> {terms.timestamp}</div>
            <div><strong>Assessment Engine:</strong> Logistic ML v1.2</div>
        </div>
    </div>

    <div class="decision-banner">
        <div>
            <div class="decision-label">Official Committee Recommendation</div>
            <div class="decision-value">{terms.decision.upper()}</div>
        </div>
        <div>
            <span class="risk-badge">Rating: {terms.risk_grade} • {terms.risk_tier}</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Recommended APR</div>
            <div class="kpi-value">{terms.recommended_apr:.2f}%</div>
            <div class="kpi-sub">Risk-Adjusted Spread</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Credit Limit</div>
            <div class="kpi-value">{terms.max_safe_credit_limit:,.0f} DM</div>
            <div class="kpi-sub">Collateral/Capacity Cap</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Monthly Amortization</div>
            <div class="kpi-value">{terms.monthly_payment_requested:,.2f} DM</div>
            <div class="kpi-sub">{terms.requested_duration} mos @ {terms.recommended_apr:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Default Probability</div>
            <div class="kpi-value">{bad_probability:.1%}</div>
            <div class="kpi-sub">Policy Cutoff: {threshold:.0%}</div>
        </div>
    </div>

    <div class="section-title">Loan Structure & Cost Comparison</div>
    <table class="table-container">
        <thead>
            <tr>
                <th>Financing Parameter</th>
                <th>Requested Application</th>
                <th>Recommended Underwriting Offer</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Principal Facility Amount</strong></td>
                <td>{terms.requested_amount:,.0f} DM</td>
                <td><strong>{terms.max_safe_credit_limit:,.0f} DM</strong> (Max Recommended)</td>
            </tr>
            <tr>
                <td><strong>Loan Tenure (Maturity)</strong></td>
                <td>{terms.requested_duration} months</td>
                <td>{min(terms.requested_duration, 36) if terms.decision != 'Approved' else terms.requested_duration} months</td>
            </tr>
            <tr>
                <td><strong>Annual Percentage Rate (APR)</strong></td>
                <td>{terms.recommended_apr:.2f}% (Risk Indexed)</td>
                <td>{terms.recommended_apr:.2f}%</td>
            </tr>
            <tr>
                <td><strong>Monthly Installment Payment</strong></td>
                <td><strong>{terms.monthly_payment_requested:,.2f} DM/mo</strong></td>
                <td><strong>{terms.monthly_payment_recommended:,.2f} DM/mo</strong></td>
            </tr>
            <tr>
                <td><strong>Total Interest Payable</strong></td>
                <td>{terms.total_interest_requested:,.2f} DM</td>
                <td>{terms.total_interest_recommended:,.2f} DM</td>
            </tr>
            <tr>
                <td><strong>Total Facility Outflow</strong></td>
                <td>{terms.total_cost_requested:,.2f} DM</td>
                <td>{terms.total_cost_recommended:,.2f} DM</td>
            </tr>
            <tr>
                <td><strong>Debt Service Ratio</strong></td>
                <td colspan="2">{terms.debt_service_burden}</td>
            </tr>
        </tbody>
    </table>

    <div class="section-title">Applicant Financial Profile Breakdown</div>
    <table class="table-container">
        <tbody>
            <tr>
                <td style="width: 25%;"><strong>Checking Status:</strong></td>
                <td style="width: 25%;">{fmt('checking_account')}</td>
                <td style="width: 25%;"><strong>Savings Balance:</strong></td>
                <td style="width: 25%;">{fmt('savings')}</td>
            </tr>
            <tr>
                <td><strong>Credit History:</strong></td>
                <td>{fmt('credit_history')}</td>
                <td><strong>Employment Tenure:</strong></td>
                <td>{fmt('employment')}</td>
            </tr>
            <tr>
                <td><strong>Primary Asset / Property:</strong></td>
                <td>{fmt('property')}</td>
                <td><strong>Housing Arrangement:</strong></td>
                <td>{fmt('housing')}</td>
            </tr>
            <tr>
                <td><strong>Purpose of Loan:</strong></td>
                <td>{fmt('purpose')}</td>
                <td><strong>Job Category:</strong></td>
                <td>{fmt('job')}</td>
            </tr>
            <tr>
                <td><strong>Age:</strong></td>
                <td>{applicant.get('age', 'N/A')} years</td>
                <td><strong>Existing Bank Credits:</strong></td>
                <td>{applicant.get('existing_credits', 1)} facility(ies)</td>
            </tr>
            <tr>
                <td><strong>Guarantors / Debtors:</strong></td>
                <td>{fmt('other_debtors')}</td>
                <td><strong>Foreign Worker:</strong></td>
                <td>{fmt('foreign_worker')}</td>
            </tr>
        </tbody>
    </table>

    <div class="section-title">Underwriting Covenants & Conditions Precedent</div>
    <ul class="conditions-list">
        {conditions_li}
    </ul>

    <div class="signature-grid">
        <div>
            <div class="sign-line"></div>
            <div class="sign-title">Lead Credit Analyst / Underwriter</div>
            <div style="font-size: 11px; color: #64748b;">Risk Analytics Division</div>
        </div>
        <div>
            <div class="sign-line"></div>
            <div class="sign-title">Chief Credit Officer / Committee Chair</div>
            <div style="font-size: 11px; color: #64748b;">Credit Committee Approval Authority</div>
        </div>
    </div>

    <div class="disclaimer">
        CONFIDENTIAL CREDIT MEMORANDUM: This document and the predictive analytics contained herein are generated by CreditIQ automated statistical risk models. The risk score and recommended parameters reflect econometric probabilities trained on historical credit performance and do not constitute an irrevocable binding loan agreement until finalized by designated lending officers.
    </div>
</div>

</body>
</html>
"""
    return html


def generate_credit_memo_json(
    applicant: Dict[str, Any],
    terms: UnderwritingTerms,
    bad_probability: float,
    threshold: float
) -> str:
    """Generates structured JSON representation for audit trails and API export."""
    payload = {
        "docket_id": terms.docket_id,
        "timestamp": terms.timestamp,
        "applicant_profile": applicant,
        "risk_evaluation": {
            "predicted_default_probability": round(bad_probability, 4),
            "policy_decision_threshold": round(threshold, 4),
            "underwriting_decision": terms.decision,
            "credit_rating_grade": terms.risk_grade,
            "risk_tier": terms.risk_tier,
        },
        "recommended_loan_terms": {
            "recommended_apr_percentage": terms.recommended_apr,
            "max_safe_credit_limit_dm": terms.max_safe_credit_limit,
            "requested_terms": {
                "principal_dm": terms.requested_amount,
                "duration_months": terms.requested_duration,
                "monthly_payment_dm": terms.monthly_payment_requested,
                "total_interest_dm": terms.total_interest_requested,
                "total_payment_dm": terms.total_cost_requested
            },
            "recommended_terms": {
                "principal_dm": min(terms.requested_amount, terms.max_safe_credit_limit),
                "duration_months": min(terms.requested_duration, 36) if terms.decision != "Approved" else terms.requested_duration,
                "monthly_payment_dm": terms.monthly_payment_recommended,
                "total_interest_dm": terms.total_interest_recommended,
                "total_payment_dm": terms.total_cost_recommended
            },
            "debt_service_burden": terms.debt_service_burden
        },
        "underwriting_conditions": terms.conditions
    }
    return json.dumps(payload, indent=2)
