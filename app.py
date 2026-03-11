"""
PhishGuard AI - Advanced Flask Backend
Features: SQLite DB, SHAP, WHOIS, Email Alerts, Hybrid Detection
Author: Vanshika Mittal | BTech/BCA Final Year Project
"""
from flask import Flask, request, jsonify, render_template_string
import pickle, json, re, os, sqlite3
from datetime import datetime
import numpy as np
from urllib.parse import urlparse

app = Flask(__name__, static_folder='static')

# ── Email config (fill your Gmail details) ──
EMAIL_CONFIG = {
    'enabled':   False,          # Set True to enable email alerts
    'sender':    'your_gmail@gmail.com',
    'password':  'your_app_password',   # Gmail App Password (not main password)
    'receiver':  'admin@gmail.com',
}

# ── Load ML artifacts ──
with open('model/best_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)
with open('model/feature_cols.json') as f:
    FEATURE_COLS = json.load(f)
with open('model/results.json') as f:
    RESULTS = json.load(f)

# ── Try loading SHAP (optional) ──
try:
    import shap
    SHAP_AVAILABLE = True
    # Build explainer once at startup
    try:
        EXPLAINER = shap.TreeExplainer(MODEL.named_steps['clf'])
    except Exception:
        SHAP_AVAILABLE = False
        EXPLAINER = None
except ImportError:
    SHAP_AVAILABLE = False
    EXPLAINER = None

# ── Try loading WHOIS (optional) ──
try:
    import whois as whois_lib
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PHISHING_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
    '.click', '.link', '.work', '.ru', '.cn', '.pw',
    '.cc', '.ws', '.zip', '.mov'
}

SUSPICIOUS_WORDS = [
    'login', 'signin', 'verify', 'secure', 'account',
    'update', 'confirm', 'banking', 'paypal', 'ebay',
    'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'netflix', 'instagram', 'whatsapp', 'support', 'helpdesk',
    'suspend', 'validate', 'recover', 'unlock', 'alert'
]

LEGIT_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
    'wikipedia.org', 'reddit.com', 'amazon.com', 'microsoft.com',
    'apple.com', 'netflix.com', 'spotify.com', 'dropbox.com',
    'zoom.us', 'slack.com', 'notion.so', 'medium.com'
}

# ── Real Government / Official domains ──
REAL_GOVT_DOMAINS = {
    'parivahan.gov.in', 'mparivahan.nic.in',
    'incometax.gov.in', 'efiling.incometax.gov.in',
    'irctc.co.in', 'indianrail.gov.in',
    'uidai.gov.in', 'myaadhaar.uidai.gov.in',
    'epfindia.gov.in', 'unifiedportal-mem.epfindia.gov.in',
    'sbi.co.in', 'onlinesbi.sbi',
    'rbi.org.in', 'npci.org.in',
    'india.gov.in', 'mca.gov.in',
    'gst.gov.in', 'cbic.gov.in',
    'passport.gov.in', 'indianvisaonline.gov.in',
}

# ── Fake / Impersonating domains (known scam sites) ──
FAKE_GOVT_DOMAINS = {
    # Parivahan fakes
    'mparivahan.org', 'mparivahan.com', 'mparivahan.in',
    'parivahan.org', 'parivahan.com', 'e-parivahan.com',
    # Income tax fakes
    'incometax.org', 'incometaxindia.org', 'efiling-incometax.com',
    # IRCTC fakes
    'irctc.org', 'irctc.com', 'irctcticket.com', 'irctc-booking.com',
    # Aadhaar fakes
    'uidai.com', 'uidai.org', 'myaadhaar.com', 'aadharcard.com',
    # EPF fakes
    'epfo.org', 'epfindia.com', 'epf-india.com',
    # Bank fakes
    'sbi.org', 'sbi.com', 'sbionline.com', 'onlinesbi.com',
    'hdfcbank.org', 'icicbank.com', 'axisbanks.com',
    # RBI fakes
    'rbi.com', 'rbi.org', 'rbigov.com',
    # GST fakes
    'gst.org', 'gstindia.com', 'gst-portal.com',
}

# ── Brand → Real domain mapping (for spoof detection) ──
BRAND_REAL_DOMAIN = {
    'parivahan':  'parivahan.gov.in',
    'mparivahan': 'mparivahan.nic.in',
    'irctc':      'irctc.co.in',
    'uidai':      'uidai.gov.in',
    'aadhaar':    'uidai.gov.in',
    'epfo':       'epfindia.gov.in',
    'incometax':  'incometax.gov.in',
    'passport':   'passport.gov.in',
    'sbi':        'sbi.co.in',
    'hdfc':       'hdfcbank.com',
    'icici':      'icicibank.com',
    'axis':       'axisbank.com',
    'rbi':        'rbi.org.in',
    'npci':       'npci.org.in',
    'gst':        'gst.gov.in',
    'paypal':     'paypal.com',
    'amazon':     'amazon.com',
    'apple':      'apple.com',
    'microsoft':  'microsoft.com',
    'google':     'google.com',
    'netflix':    'netflix.com',
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

DB_PATH = 'database/phishguard.db'

def init_db():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_history (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        url              TEXT NOT NULL,
        prediction       INTEGER NOT NULL,
        label            TEXT NOT NULL,
        confidence       REAL NOT NULL,
        risk_factors     TEXT,
        detection_method TEXT,
        ml_confidence    REAL,
        domain_age_days  INTEGER,
        shap_top_feature TEXT,
        scanned_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        id          INTEGER PRIMARY KEY,
        total_scans INTEGER DEFAULT 0,
        phish_count INTEGER DEFAULT 0,
        legit_count INTEGER DEFAULT 0
    )''')
    c.execute('SELECT COUNT(*) FROM stats')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO stats VALUES (1,0,0,0)')
    conn.commit()
    conn.close()

def save_scan(url, prediction, label, confidence,
              risk_factors, detection_method, ml_confidence,
              domain_age_days=None, shap_top_feature=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO scan_history
        (url,prediction,label,confidence,risk_factors,detection_method,
         ml_confidence,domain_age_days,shap_top_feature)
        VALUES (?,?,?,?,?,?,?,?,?)''',
        (url, prediction, label, confidence,
         json.dumps(risk_factors), detection_method,
         ml_confidence, domain_age_days, shap_top_feature))
    c.execute('''UPDATE stats SET
        total_scans = total_scans + 1,
        phish_count = phish_count + ?,
        legit_count = legit_count + ?
        WHERE id = 1
    ''',
        (1 if prediction == 1 else 0,
         0 if prediction == 1 else 1))
    conn.commit()
    conn.close()

def get_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT url,label,confidence,risk_factors,
                        detection_method,domain_age_days,scanned_at
                 FROM scan_history ORDER BY id DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [{
        'url': r[0], 'label': r[1], 'confidence': r[2],
        'risk_factors':     json.loads(r[3]) if r[3] else [],
        'detection_method': r[4],
        'domain_age_days':  r[5],
        'scanned_at':       r[6]
    } for r in rows]

def get_db_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT total_scans,phish_count,legit_count FROM stats WHERE id=1')
    row = c.fetchone()
    conn.close()
    return {'total': row[0], 'phish': row[1], 'legit': row[2]} if row else {'total':0,'phish':0,'legit':0}

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(url):
    parsed   = urlparse(url if url.startswith('http') else 'http://' + url)
    hostname = parsed.netloc or parsed.path
    path     = parsed.path
    features = {
        'url_length':           len(url),
        'num_dots':             url.count('.'),
        'num_hyphens':          url.count('-'),
        'num_underscores':      url.count('_'),
        'num_slash':            url.count('/'),
        'num_question':         url.count('?'),
        'num_equals':           url.count('='),
        'num_at':               url.count('@'),
        'num_ampersand':        url.count('&'),
        'num_digits':           sum(c.isdigit() for c in url),
        'digit_ratio':          sum(c.isdigit() for c in url) / max(len(url), 1),
        'hostname_length':      len(hostname),
        'has_ip':               1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0,
        'subdomain_count':      max(0, len(hostname.split('.')) - 2),
        'has_https':            1 if parsed.scheme == 'https' else 0,
        'suspicious_words':     sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        'path_length':          len(path),
        'num_path_components':  len([p for p in path.split('/') if p]),
        'has_port':             1 if parsed.port else 0,
        'tld_in_path':          1 if re.search(r'\.(com|net|org|edu)', path) else 0,
    }
    return [features[col] for col in FEATURE_COLS]

# ─────────────────────────────────────────────
# WHOIS DOMAIN AGE
# ─────────────────────────────────────────────

def get_domain_age(hostname):
    """Returns domain age in days, or None if unavailable."""
    if not WHOIS_AVAILABLE:
        return None
    try:
        parts = hostname.split('.')
        domain = '.'.join(parts[-2:]) if len(parts) >= 2 else hostname
        w = whois_lib.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            age = (datetime.now() - created).days
            return age
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────

def get_shap_explanation(feats):
    """Returns top SHAP feature contributions for this prediction."""
    if not SHAP_AVAILABLE or EXPLAINER is None:
        # Fallback: use feature importance
        return get_fallback_explanation(feats)
    try:
        scaler = MODEL.named_steps['scaler']
        X_scaled = scaler.transform([feats])
        shap_values = EXPLAINER.shap_values(X_scaled)
        # For binary classification, shap_values[1] = phishing class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        # Top 5 features by absolute SHAP value
        top_idx = np.argsort(np.abs(sv))[-5:][::-1]
        explanations = []
        for i in top_idx:
            feat_name  = FEATURE_COLS[i]
            feat_val   = round(feats[i], 3)
            shap_val   = round(float(sv[i]), 4)
            direction  = 'PHISHING' if shap_val > 0 else 'LEGIT'
            explanations.append({
                'feature':   feat_name,
                'value':     feat_val,
                'shap':      shap_val,
                'direction': direction,
                'impact':    abs(shap_val)
            })
        return explanations
    except Exception:
        return get_fallback_explanation(feats)

def get_fallback_explanation(feats):
    """Simple rule-based explanation when SHAP unavailable."""
    feat_dict = dict(zip(FEATURE_COLS, feats))
    explanations = []
    checks = [
        ('has_ip',           'IP as hostname',         0.42),
        ('digit_ratio',      'High digit ratio',        0.31),
        ('suspicious_words', 'Suspicious keywords',     0.28),
        ('url_length',       'URL length',              0.18),
        ('subdomain_count',  'Subdomain count',         0.17),
        ('has_https',        'HTTPS absent',            0.15),
        ('num_hyphens',      'Hyphen count',            0.12),
        ('tld_in_path',      'TLD in path',             0.10),
    ]
    for feat, label, base_impact in checks[:5]:
        val = feat_dict.get(feat, 0)
        explanations.append({
            'feature':   feat,
            'value':     round(val, 3),
            'shap':      round(base_impact * (val if val <= 1 else 0.5), 4),
            'direction': 'PHISHING' if val > 0 else 'LEGIT',
            'impact':    base_impact
        })
    return explanations

# ─────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────

def rule_based_check(url, hostname, parsed):
    url_lower = url.lower()
    reasons, phish_score = [], 0
    parts = hostname.split('.')

    # Handle multi-part TLDs like .gov.in, .co.in, .nic.in
    if len(parts) >= 3 and parts[-2] in ('gov','co','nic','org','net'):
        actual_domain = '.'.join(parts[-3:])
    else:
        actual_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else hostname

    # ── Rule 0: Known fake govt / impersonation domains ──
    if actual_domain in FAKE_GOVT_DOMAINS:
        reasons.append(f'Known fake/scam domain "{actual_domain}" — impersonating official site')
        return True, 98.0, reasons

    # ── Rule 0b: Brand → real domain spoof check ──
    for brand, real_domain in BRAND_REAL_DOMAIN.items():
        if brand in url_lower:
            # Extract real domain's base
            real_base = '.'.join(real_domain.split('.')[-2:]) if '.' in real_domain else real_domain
            # Check if current domain is NOT the real domain
            if real_base not in actual_domain and actual_domain not in real_domain:
                reasons.append(f'"{brand}" brand used but domain is "{actual_domain}" (real: {real_domain})')
                phish_score += 55
                break

    for tld in PHISHING_TLDS:
        if hostname.endswith(tld):
            reasons.append(f'Suspicious free TLD ({tld}) — massively used in phishing')
            phish_score += 45; break

    brands = ['paypal','amazon','apple','microsoft','google','facebook',
              'netflix','ebay','instagram','bank','flipkart','hdfc','sbi']
    for brand in brands:
        if brand in url_lower and brand not in actual_domain:
            reasons.append(f'Brand "{brand}" impersonated in subdomain')
            phish_score += 40; break

    sw_count = sum(w in url_lower for w in SUSPICIOUS_WORDS)
    if sw_count >= 3:
        found = [w for w in SUSPICIOUS_WORDS if w in url_lower][:4]
        reasons.append(f'Multiple phishing keywords: {", ".join(found)}')
        phish_score += 30

    if re.match(r'\d+\.\d+\.\d+\.\d+', hostname):
        reasons.append('IP address as hostname — legitimate sites use domain names')
        phish_score += 45

    if '@' in url:
        reasons.append('@ symbol in URL — classic redirect trick')
        phish_score += 35

    if len(parts) - 2 >= 3:
        reasons.append(f'Excessive subdomains ({len(parts)-2})')
        phish_score += 20

    if parsed.scheme != 'https' and sw_count >= 2:
        reasons.append('No HTTPS on sensitive page')
        phish_score += 20

    if len(url) > 100:
        reasons.append(f'Unusually long URL ({len(url)} chars)')
        phish_score += 10

    if actual_domain.count('-') >= 2:
        reasons.append('Multiple hyphens in domain name')
        phish_score += 15

    if re.search(r'\.(com|net|org|edu|gov)', parsed.path):
        reasons.append('Legitimate TLD found inside URL path')
        phish_score += 15

    for legit in LEGIT_DOMAINS:
        if actual_domain == legit:
            return False, 2.0, []

    # Real government domains are always safe
    for real_gov in REAL_GOVT_DOMAINS:
        if hostname == real_gov or hostname.endswith('.' + real_gov):
            return False, 2.0, []

    if phish_score >= 40:
        return True, round(min(99.0, 55.0 + phish_score * 0.45), 1), reasons
    elif phish_score >= 15:
        return None, phish_score, reasons
    return None, 0, reasons

# ─────────────────────────────────────────────
# EMAIL ALERT
# ─────────────────────────────────────────────

def send_email_alert(url, confidence, risk_factors):
    if not EMAIL_CONFIG['enabled']:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From']    = EMAIL_CONFIG['sender']
        msg['To']      = EMAIL_CONFIG['receiver']
        msg['Subject'] = f'⚠️ PhishGuard Alert: Phishing Detected ({confidence}% confidence)'

        body = f"""
PhishGuard AI — Phishing Alert
================================
URL        : {url}
Confidence : {confidence}%
Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Risk Factors:
{chr(10).join(f'  • {r}' for r in risk_factors)}

-- PhishGuard AI System
        """
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['sender'], EMAIL_CONFIG['receiver'], msg.as_string())
        print(f"📧 Alert email sent for: {url}")
    except Exception as e:
        print(f"📧 Email failed: {e}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    html = open('templates/index.html', encoding='utf-8').read()
    html = html.replace('"__RESULTS_DATA__"', json.dumps(RESULTS))
    return html

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    parsed   = urlparse(url if url.startswith('http') else 'http://' + url)
    hostname = parsed.netloc or parsed.path

    # ── Rule engine ──
    rule_result, rule_conf, rule_reasons = rule_based_check(url, hostname, parsed)

    # ── ML model ──
    feats   = extract_features(url)
    X       = np.array([feats])
    ml_pred = int(MODEL.predict(X)[0])
    ml_prob = float(MODEL.predict_proba(X)[0][1])

    # ── Combine ──
    if rule_result is True:
        final_pred, final_prob = 1, rule_conf / 100
        detection_method = 'RULE + ML HYBRID'
    elif rule_result is False:
        final_pred, final_prob = 0, 0.02
        detection_method = 'WHITELIST'
    else:
        boost        = (rule_conf / 100) * 0.35
        boosted_prob = min(0.99, ml_prob + boost)
        final_pred   = 1 if boosted_prob >= 0.5 else 0
        final_prob   = boosted_prob
        detection_method = 'ML MODEL'

    # ── Risk factors ──
    feat_dict    = dict(zip(FEATURE_COLS, feats))
    risk_factors = list(rule_reasons)
    rf_str = ' '.join(risk_factors).lower()

    if feat_dict['has_ip']          and 'ip'       not in rf_str:
        risk_factors.append('IP address as hostname')
    if feat_dict['num_at'] > 0      and '@'        not in rf_str:
        risk_factors.append('@ symbol detected')
    if feat_dict['url_length'] > 75 and 'long url' not in rf_str:
        risk_factors.append(f"Long URL ({int(feat_dict['url_length'])} chars)")
    if feat_dict['suspicious_words'] > 0 and 'keyword' not in rf_str:
        risk_factors.append('Suspicious keywords in URL')
    if not feat_dict['has_https']   and 'https'    not in rf_str:
        risk_factors.append('No HTTPS protocol')
    if feat_dict['subdomain_count'] > 2 and 'subdomain' not in rf_str:
        risk_factors.append(f"Excessive subdomains ({int(feat_dict['subdomain_count'])})")
    if feat_dict['tld_in_path']     and 'tld'      not in rf_str:
        risk_factors.append('TLD found inside URL path')
    risk_factors = list(dict.fromkeys(risk_factors))

    confidence    = round(final_prob * 100, 1)
    ml_confidence = round(ml_prob   * 100, 1)

    # ── WHOIS domain age ──
    domain_age_days = get_domain_age(hostname)
    domain_info = {}
    if domain_age_days is not None:
        domain_info['age_days'] = domain_age_days
        domain_info['age_label'] = (
            f'{domain_age_days} days old'
            if domain_age_days < 365
            else f'{domain_age_days // 365} year(s) old'
        )
        domain_info['suspicious'] = domain_age_days < 30
        if domain_age_days < 30 and final_pred == 1:
            risk_factors.append(f'Very new domain ({domain_age_days} days old) — common phishing pattern')

    # ── SHAP explanation ──
    shap_explanation = get_shap_explanation(feats)
    shap_top = shap_explanation[0]['feature'] if shap_explanation else None

    # ── Save to DB ──
    save_scan(url, final_pred,
              'PHISHING' if final_pred == 1 else 'LEGITIMATE',
              confidence, risk_factors, detection_method,
              ml_confidence, domain_age_days, shap_top)

    # ── Email alert ──
    if final_pred == 1 and confidence >= 80:
        send_email_alert(url, confidence, risk_factors)

    return jsonify({
        'prediction':        final_pred,
        'label':             'PHISHING' if final_pred == 1 else 'LEGITIMATE',
        'confidence':        confidence,
        'risk_factors':      risk_factors,
        'features':          {k: round(v, 3) for k, v in feat_dict.items()},
        'detection_method':  detection_method,
        'ml_confidence':     ml_confidence,
        'domain_info':       domain_info,
        'shap_explanation':  shap_explanation,
        'shap_available':    SHAP_AVAILABLE,
        'whois_available':   WHOIS_AVAILABLE,
    })

@app.route('/history')
def history():
    limit = request.args.get('limit', 100, type=int)
    return jsonify(get_history(limit))

@app.route('/db-stats')
def db_stats():
    return jsonify(get_db_stats())

@app.route('/model-stats')
def model_stats():
    return jsonify(RESULTS)

# ─────────────────────────────────────────────
# EMAIL / SMS SCAM DETECTION ENGINE
# ─────────────────────────────────────────────

URGENCY_WORDS = [
    'urgent', 'immediately', 'act now', 'limited time', 'expires',
    'suspended', 'blocked', 'locked', 'verify now', 'confirm now',
    'last chance', 'final notice', 'account disabled', 'unusual activity',
    'security alert', 'warning', 'action required', 'respond immediately',
    'within 24 hours', 'within 48 hours', 'your account will be',
    'click here immediately', 'do not ignore'
]

REWARD_WORDS = [
    'congratulations', 'you have won', 'winner', 'prize', 'reward',
    'free gift', 'claim now', 'selected', 'lucky winner', 'cash prize',
    'lottery', 'jackpot', 'you are selected', 'exclusive offer',
    'special offer', 'limited offer', 'bonus', 'gift card', 'voucher'
]

THREAT_WORDS = [
    'legal action', 'police', 'arrested', 'lawsuit', 'court',
    'fbi', 'irs', 'income tax', 'cybercrime', 'hacked', 'compromised',
    'virus detected', 'malware', 'your device', 'criminal charges',
    'fine', 'penalty', 'warrant'
]

FINANCIAL_WORDS = [
    'bank account', 'credit card', 'debit card', 'atm', 'otp',
    'pin number', 'cvv', 'account number', 'ifsc', 'swift',
    'wire transfer', 'send money', 'payment required', 'outstanding amount',
    'overdue', 'refund', 'transaction failed', 'kyc', 'aadhar', 'pan card'
]

SENSITIVE_REQUEST = [
    'password', 'username', 'social security', 'date of birth',
    'mother maiden', 'security question', 'secret answer',
    'enter your details', 'fill in your', 'provide your',
    'share your', 'send us your'
]

SCAM_BRANDS = [
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'netflix', 'ebay', 'bank of america', 'chase', 'wells fargo',
    'hdfc', 'sbi', 'icici', 'axis bank', 'paytm', 'phonepe',
    'flipkart', 'myntra', 'irctc', 'uidai', 'income tax department',
    'rbi', 'epfo', 'lic', 'jio', 'airtel', 'vodafone'
]

def extract_urls_from_text(text):
    """Extract all URLs from text/email."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    bare_pattern = r'\b(?:www\.)[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    urls += re.findall(bare_pattern, text, re.IGNORECASE)
    return list(set(urls))

def analyze_text_scam(text):
    """Full scam analysis of email/SMS text."""
    text_lower = text.lower()
    score = 0
    reasons = []
    categories = []

    # 1. Urgency detection
    urg_found = [w for w in URGENCY_WORDS if w in text_lower]
    if len(urg_found) >= 2:
        score += 35
        reasons.append(f'High urgency language: "{urg_found[0]}", "{urg_found[1]}"')
        categories.append('URGENCY')
    elif len(urg_found) == 1:
        score += 18
        reasons.append(f'Urgency language detected: "{urg_found[0]}"')
        categories.append('URGENCY')

    # 2. Reward/lottery scam
    rew_found = [w for w in REWARD_WORDS if w in text_lower]
    if rew_found:
        score += 40
        reasons.append(f'Reward/lottery scam words: "{rew_found[0]}"')
        categories.append('REWARD SCAM')

    # 3. Threat/fear tactics
    thr_found = [w for w in THREAT_WORDS if w in text_lower]
    if thr_found:
        score += 38
        reasons.append(f'Fear/threat tactics: "{thr_found[0]}"')
        categories.append('THREAT SCAM')

    # 4. Financial info request
    fin_found = [w for w in FINANCIAL_WORDS if w in text_lower]
    if len(fin_found) >= 2:
        score += 42
        reasons.append(f'Requests financial info: {", ".join(fin_found[:3])}')
        categories.append('FINANCIAL FRAUD')
    elif len(fin_found) == 1:
        score += 20
        reasons.append(f'Financial keyword: "{fin_found[0]}"')

    # 5. Sensitive info request
    sen_found = [w for w in SENSITIVE_REQUEST if w in text_lower]
    if sen_found:
        score += 35
        reasons.append(f'Requests sensitive info: "{sen_found[0]}"')
        categories.append('CREDENTIAL THEFT')

    # 6. Brand impersonation
    brand_found = [b for b in SCAM_BRANDS if b in text_lower]
    if brand_found:
        score += 20
        reasons.append(f'Brand impersonation: "{brand_found[0]}"')
        categories.append('BRAND IMPERSONATION')

    # 7. URLs in message
    urls = extract_urls_from_text(text)
    suspicious_urls = []
    if urls:
        score += 15
        reasons.append(f'{len(urls)} URL(s) found in message')
        # Quick check each URL
        for url in urls[:3]:
            parsed = urlparse(url if url.startswith('http') else 'http://' + url)
            hostname = parsed.netloc or parsed.path
            parts = hostname.split('.')
            tld = '.' + parts[-1] if parts else ''
            if tld in PHISHING_TLDS:
                score += 25
                suspicious_urls.append(url)
                reasons.append(f'Suspicious URL with free TLD: {url[:60]}')
            if any(b in hostname.lower() for b in ['paypal','amazon','apple','bank','secure']):
                score += 20
                suspicious_urls.append(url)
                reasons.append(f'Suspicious brand URL: {url[:60]}')

    # 8. ALL CAPS words (shouting)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    caps_words = [w for w in caps_words if w not in ['HTTP','HTTPS','HTML','FROM','DEAR']]
    if len(caps_words) >= 3:
        score += 12
        reasons.append(f'Excessive CAPS words: {", ".join(caps_words[:4])}')

    # 9. Grammar/spelling indicators (simple check)
    grammar_red_flags = ['kindly do the needful', 'do the needful', 'revert back',
                         'dear customer', 'dear user', 'dear sir/madam',
                         'valued customer', 'esteemed customer']
    gram_found = [g for g in grammar_red_flags if g in text_lower]
    if gram_found:
        score += 10
        reasons.append(f'Generic/suspicious greeting: "{gram_found[0]}"')

    # 10. Message length check (very short = smishing)
    word_count = len(text.split())
    if word_count < 20 and urls:
        score += 15
        reasons.append('Very short message with URL — classic SMS phishing (smishing)')
        categories.append('SMISHING')

    # Final verdict
    score = min(score, 99)
    if score >= 60:
        label = 'SCAM'
        risk  = 'HIGH' if score >= 80 else 'MEDIUM'
    elif score >= 30:
        label = 'SUSPICIOUS'
        risk  = 'MEDIUM'
    else:
        label = 'SAFE'
        risk  = 'LOW'

    return {
        'label':        label,
        'risk':         risk,
        'score':        score,
        'reasons':      reasons,
        'categories':   list(set(categories)),
        'urls_found':   urls,
        'suspicious_urls': suspicious_urls,
        'word_count':   word_count,
        'urgency_count':    len(urg_found),
        'financial_count':  len(fin_found),
        'threat_count':     len(thr_found),
    }

@app.route('/scan-text', methods=['POST'])
def scan_text():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) > 10000:
        return jsonify({'error': 'Text too long (max 10000 chars)'}), 400

    result = analyze_text_scam(text)

    # Also scan any URLs found inside the message
    url_scan_results = []
    for url in result['urls_found'][:3]:
        try:
            parsed   = urlparse(url if url.startswith('http') else 'http://' + url)
            hostname = parsed.netloc or parsed.path
            rule_result, rule_conf, rule_reasons = rule_based_check(url, hostname, parsed)
            feats    = extract_features(url)
            ml_prob  = float(MODEL.predict_proba(np.array([feats]))[0][1])
            url_scan_results.append({
                'url':        url,
                'phishing':   rule_result is True or ml_prob > 0.5,
                'confidence': round((rule_conf if rule_result is True else ml_prob * 100), 1)
            })
        except Exception:
            pass

    result['url_scan_results'] = url_scan_results
    return jsonify(result)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    print("=" * 50)
    print("  ✅  PhishGuard AI — Advanced Edition")
    print("=" * 50)
    print(f"  📊  Dashboard  : http://localhost:5000")
    print(f"  🗄️   Database  : {DB_PATH}")
    print(f"  🤖  SHAP       : {'✅ Available' if SHAP_AVAILABLE else '❌ Not available (pip install shap)'}")
    print(f"  🌐  WHOIS      : {'✅ Available' if WHOIS_AVAILABLE else '❌ Not available (pip install python-whois)'}")
    print(f"  📧  Email      : {'✅ Enabled' if EMAIL_CONFIG['enabled'] else '❌ Disabled (set enabled=True in config)'}")
    print("=" * 50)
    app.run(debug=True, port=5000)