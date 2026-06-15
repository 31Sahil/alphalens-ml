import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import StringIO
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, auc,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AlphaLens ML — Indian Markets",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg0:#03091A;--bg1:#07112E;--bg2:#0D1B40;--glass:rgba(255,255,255,0.035);--glass2:rgba(255,255,255,0.06);--border:rgba(255,255,255,0.08);--border2:rgba(0,201,255,0.25);--cyan:#00C9FF;--purple:#8B5CF6;--green:#10B981;--red:#F43F5E;--amber:#F59E0B;--text1:#F0F4FF;--text2:#8B9DC3;--text3:#4A5A80;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg0)!important;font-family:'Inter',sans-serif;color:var(--text1);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--bg1) 0%,var(--bg0) 100%)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text1)!important;}
.ml-header{background:linear-gradient(135deg,rgba(0,201,255,0.09) 0%,rgba(139,92,246,0.09) 100%);border:1px solid var(--border2);border-radius:14px;padding:24px 32px;margin-bottom:24px;position:relative;overflow:hidden;}
.ml-header::before{content:'';position:absolute;top:-50px;right:-50px;width:180px;height:180px;background:radial-gradient(circle,rgba(0,201,255,0.13) 0%,transparent 70%);border-radius:50%;}
.header-title{font-size:26px;font-weight:800;background:linear-gradient(135deg,var(--cyan) 0%,var(--purple) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 4px 0;}
.header-sub{font-size:13px;color:var(--text2);margin:0;}
[data-testid="stTabs"] [role="tablist"]{background:var(--bg1);border-radius:14px;padding:4px;border:1px solid var(--border);}
[data-testid="stTabs"] [role="tab"]{background:transparent!important;color:var(--text2)!important;border-radius:8px!important;font-size:12px!important;font-weight:500!important;padding:8px 16px!important;border:none!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:linear-gradient(135deg,rgba(0,201,255,0.18),rgba(139,92,246,0.18))!important;color:var(--text1)!important;border:1px solid var(--border2)!important;}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px;}
.kpi-card{background:var(--glass);border:1px solid var(--border);border-radius:8px;padding:14px 16px;position:relative;overflow:hidden;}
.kpi-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--c,rgba(0,201,255,0.5));}
.kpi-label{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:4px;}
.kpi-value{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--text1);}
.kpi-sub{font-size:10px;color:var(--text3);margin-top:3px;}
.sec-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--text2);margin:20px 0 10px 0;display:flex;align-items:center;gap:8px;}
.sec-title::after{content:'';flex:1;height:1px;background:var(--border);}
.insight-card{background:var(--glass);border:1px solid var(--border);border-left:3px solid var(--cyan);border-radius:8px;padding:12px 16px;margin:8px 0;font-size:12px;color:var(--text2);}
.insight-card b{color:var(--text1);}
.badge{display:inline-block;padding:3px 9px;border-radius:5px;font-size:11px;font-weight:600;}
.badge-green{background:rgba(16,185,129,0.15);color:#10B981;border:1px solid rgba(16,185,129,0.3);}
.badge-red{background:rgba(244,63,94,0.12);color:#F43F5E;border:1px solid rgba(244,63,94,0.3);}
.badge-amber{background:rgba(245,158,11,0.12);color:#F59E0B;border:1px solid rgba(245,158,11,0.3);}
.badge-cyan{background:rgba(0,201,255,0.12);color:#00C9FF;border:1px solid rgba(0,201,255,0.3);}
.stButton>button{background:linear-gradient(135deg,rgba(0,201,255,0.15),rgba(139,92,246,0.15))!important;border:1px solid var(--border2)!important;color:var(--text1)!important;border-radius:8px!important;font-weight:600!important;}
.stButton>button:hover{background:linear-gradient(135deg,rgba(0,201,255,0.28),rgba(139,92,246,0.28))!important;}
[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--border);border-radius:8px;padding:14px!important;}
[data-testid="stMetricLabel"]{font-size:10px!important;color:var(--text2)!important;text-transform:uppercase;}
[data-testid="stMetricValue"]{font-size:22px!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important;}
::-webkit-scrollbar{width:4px;height:4px;}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.stProgress>div>div{background:linear-gradient(90deg,var(--cyan),var(--purple))!important;}
.footer-txt{font-size:10px;color:var(--text3);text-align:center;padding:16px 0;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CHART_BG   = '#03091A'
CHART_GRID = 'rgba(255,255,255,0.04)'
C_GREEN='#10B981'; C_RED='#F43F5E'; C_CYAN='#00C9FF'; C_PURPLE='#8B5CF6'; C_AMBER='#F59E0B'

# ── Nifty LargeMidcap 250 Universe ───────────────────────────────────────────
TICKERS = [
    # Large Cap — Nifty 100
    'HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS','AXISBANK.NS','SBIN.NS','INDUSINDBK.NS',
    'BANKBARODA.NS','PNB.NS','CANBK.NS','UNIONBANK.NS',
    'RELIANCE.NS','ONGC.NS','BPCL.NS','IOC.NS','HINDPETRO.NS','GAIL.NS','PETRONET.NS',
    'TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS','LTIM.NS','PERSISTENT.NS',
    'HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','BRITANNIA.NS','DABUR.NS','MARICO.NS',
    'COLPAL.NS','GODREJCP.NS',
    'SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVI.NS','LUPIN.NS','BIOCON.NS',
    'TORNTPHARM.NS',
    'MARUTI.NS','TATAMOTORS.NS','M&M.NS','BAJAJ-AUTO.NS','HEROMOTOCO.NS','EICHERMOT.NS',
    'MOTHERSON.NS',
    'NTPC.NS','POWERGRID.NS','TATAPOWER.NS','ADANIGREEN.NS','JSWENERGY.NS','COALINDIA.NS',
    'TATASTEEL.NS','JSWSTEEL.NS','HINDALCO.NS','VEDL.NS','SAIL.NS','NMDC.NS',
    'JINDALSTEL.NS',
    'LT.NS','ABB.NS','SIEMENS.NS','BEL.NS','BOSCHLTD.NS',
    'ASIANPAINT.NS','BERGEPAINT.NS','PIDILITIND.NS','SRF.NS','ASTRAL.NS',
    'BAJFINANCE.NS','BAJAJFINSV.NS','CHOLAFIN.NS','MUTHOOTFIN.NS','SBICARD.NS',
    'SBILIFE.NS','HDFCLIFE.NS','ICICIGI.NS','ICICIPRULI.NS',
    'BHARTIARTL.NS','ULTRACEMCO.NS','GRASIM.NS','SHREECEM.NS',
    'TITAN.NS','TRENT.NS','ADANIENT.NS','ADANIPORTS.NS','ZOMATO.NS',
    'APOLLOHOSP.NS','INDIGO.NS','DLF.NS','LICI.NS',
    # Mid Cap — Nifty Midcap 150
    'APOLLOTYRE.NS','CEATLTD.NS','MINDA.NS','SONACOMS.NS','TIINDIA.NS',
    'EXIDEIND.NS','SCHAEFFLER.NS','ENDURANCE.NS','SUPRAJIT.NS','MAHINDCIE.NS',
    'ABCAPITAL.NS','ANGELONE.NS','CANFINHOME.NS','EQUITASBNK.NS','JMFINANCL.NS',
    'LICHSGFIN.NS','MCX.NS','MOTILALOFS.NS','POONAWALLA.NS','SHRIRAMFIN.NS',
    'SUNDARMFIN.NS','UJJIVAN.NS','UTIAMC.NS','IIFL.NS','MANAPPURAM.NS',
    'RBLBANK.NS','FEDERALBNK.NS','KARURVYSYA.NS',
    'COFORGE.NS','HAPPSTMNDS.NS','KPITTECH.NS','MPHASIS.NS','TATAELXSI.NS',
    'TANLA.NS','MASTEK.NS','BIRLASOFT.NS','CYIENT.NS',
    'AUROPHARMA.NS','GLAND.NS','GRANULES.NS','LALPATHLAB.NS','METROPOLIS.NS',
    'NATCOPHARM.NS','PFIZER.NS','ALKYLAMINE.NS',
    'ABFRL.NS','BATAINDIA.NS','DEVYANI.NS','JYOTHYLAB.NS','MANYAVAR.NS',
    'RADICO.NS','VSTIND.NS',
    'AARTIIND.NS','DEEPAKFERT.NS','FINEORG.NS','FLUOROCHEM.NS','GALAXYSURF.NS',
    'GNFC.NS','NAVINFLUOR.NS','NOCIL.NS','PIIND.NS','VINATI.NS',
    'CUMMINSIND.NS','GRINDWELL.NS','KEC.NS','KNRCON.NS','NCC.NS',
    'THERMAX.NS','WELSPUNIND.NS','APLAPOLLO.NS',
    'BRIGADE.NS','GODREJPROP.NS','OBEROI.NS','PHOENIXLTD.NS','PRESTIGE.NS',
    'SOBHA.NS','SUNTECK.NS',
    'NATIONALUM.NS','RATNAMANI.NS','WELCORP.NS','MOIL.NS','GPIL.NS',
    'NAZARA.NS','PVRINOX.NS','SAREGAMA.NS',
    'CREDITACC.NS','KFINTECH.NS',
]
TICKERS = list(dict.fromkeys(TICKERS))  # ~200 stocks, deduped

FEATURE_COLS = [
    'Price_EMA20_Ratio','Price_EMA50_Ratio','Price_SMA200_Ratio',
    'EMA20_EMA50_Ratio','EMA50_SMA200_Ratio',
    'RSI_14','MACD_Ratio','MACD_Hist','ROC_5','ROC_10','ROC_21','ROC_63','Mom_6_1',
    'ATR_Pct','HV_20','BB_Pct',
    'Vol_Ratio','CMF_20','OBV_ROC',
    'ADX_14','DI_Diff','Stoch_K','Stoch_D',
    'High_6M_Pct','Low_6M_Pct',
]

def chl(fig, title='', h=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#8B9DC3', family='Inter')),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family='Inter', color='#8B9DC3', size=11),
        height=h, margin=dict(l=40,r=20,t=40 if title else 20,b=40),
        xaxis=dict(showgrid=False, zeroline=False, color='#4A5A80'),
        yaxis=dict(showgrid=True, gridcolor=CHART_GRID, zeroline=False, color='#4A5A80'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        hovermode='x unified',
    )
    return fig


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df):
    c=df['Close']; h=df['High']; l=df['Low']; v=df['Volume']
    df['EMA_20']=c.ewm(span=20,adjust=False).mean()
    df['EMA_50']=c.ewm(span=50,adjust=False).mean()
    df['SMA_200']=c.rolling(200).mean()
    df['Price_EMA20_Ratio']=c/df['EMA_20']
    df['Price_EMA50_Ratio']=c/df['EMA_50']
    df['Price_SMA200_Ratio']=c/df['SMA_200']
    df['EMA20_EMA50_Ratio']=df['EMA_20']/df['EMA_50']
    df['EMA50_SMA200_Ratio']=df['EMA_50']/df['SMA_200']
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(span=14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(span=14,adjust=False).mean()
    df['RSI_14']=100-100/(1+gain/loss.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    df['MACD']=e12-e26
    df['MACD_Sig']=df['MACD'].ewm(span=9,adjust=False).mean()
    df['MACD_Hist']=df['MACD']-df['MACD_Sig']
    df['MACD_Ratio']=df['MACD']/c.replace(0,np.nan)
    df['ROC_5']=c.pct_change(5)*100
    df['ROC_10']=c.pct_change(10)*100
    df['ROC_21']=c.pct_change(21)*100
    df['ROC_63']=c.pct_change(63)*100
    df['Mom_6_1']=c.pct_change(126)-c.pct_change(21)   # 6M-1M momentum (avoids 252-bar need)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['ATR_14']=tr.ewm(span=14,adjust=False).mean()
    df['ATR_Pct']=df['ATR_14']/c.replace(0,np.nan)*100
    df['HV_20']=c.pct_change().rolling(20).std()*np.sqrt(252)*100
    df['BB_Mid']=c.rolling(20).mean(); df['BB_Std']=c.rolling(20).std()
    df['BB_Pct']=(c-(df['BB_Mid']-2*df['BB_Std']))/(4*df['BB_Std']).replace(0,np.nan)
    df['Vol_MA20']=v.rolling(20).mean()
    df['Vol_Ratio']=v/df['Vol_MA20'].replace(0,np.nan)
    cmf_num=((c-l-(h-c))/(h-l).replace(0,np.nan)*v).rolling(20).sum()
    df['CMF_20']=cmf_num/v.rolling(20).sum().replace(0,np.nan)
    df['OBV_ROC']=((np.sign(c.diff())*v).fillna(0).cumsum()).pct_change(10)
    dmp=h.diff().clip(lower=0).where(h.diff()>l.diff().abs(),0)
    dmn=l.diff().abs().clip(lower=0).where(l.diff().abs()>h.diff(),0)
    atr_s=tr.ewm(span=14).mean()
    pdi=100*dmp.ewm(span=14).mean()/atr_s.replace(0,np.nan)
    ndi=100*dmn.ewm(span=14).mean()/atr_s.replace(0,np.nan)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    df['ADX_14']=dx.ewm(span=14).mean()
    df['DI_Diff']=pdi-ndi
    l14=l.rolling(14).min(); h14=h.rolling(14).max()
    df['Stoch_K']=100*(c-l14)/(h14-l14).replace(0,np.nan)
    df['Stoch_D']=df['Stoch_K'].rolling(3).mean()
    df['High_6M']=h.rolling(126).max()          # 6-month high (126 bars, not 252)
    df['High_6M_Pct']=c/df['High_6M']
    df['Low_6M']=l.rolling(126).min()
    df['Low_6M_Pct']=c/df['Low_6M']
    df['Fwd_1M']=c.shift(-21)/c-1
    df['Fwd_3M']=c.shift(-63)/c-1
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def load_and_build(tickers):
    END=datetime.today(); START=END-timedelta(days=730)   # 2 years — ensures 126-bar warmup + forward labels
    all_dfs={}
    for tkr in tickers:
        try:
            raw=yf.download(tkr,start=START,end=END,auto_adjust=True,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw=raw.droplevel(1,axis=1)
            if len(raw)<100: continue
            raw['Ticker']=tkr
            raw=compute_features(raw)
            all_dfs[tkr]=raw
        except: pass

    records=[]
    for tkr,df in all_dfs.items():
        sub=df[FEATURE_COLS+['Fwd_1M','Fwd_3M','Ticker']].dropna()
        sub=sub.iloc[:-25] if len(sub)>60 else sub   # drop last 25 rows (forward label unknown)
        records.append(sub)
    if not records:
        return None, None, {}

    master=pd.concat(records,ignore_index=True)
    master=master.dropna(subset=FEATURE_COLS+['Fwd_1M'])
    master=master.replace([np.inf,-np.inf],np.nan).dropna()

    BUY_THR=0.04; SELL_THR=-0.04
    master['Label']=master['Fwd_1M'].apply(
        lambda r: 2 if r>BUY_THR else (0 if r<SELL_THR else 1))
    master['Label_Name']=master['Label'].map({0:'SELL',1:'NEUTRAL',2:'BUY'})

    # Latest features per ticker
    latest={}
    for tkr,df in all_dfs.items():
        row=df[FEATURE_COLS].dropna().tail(1)
        if len(row)==1: latest[tkr]=row.iloc[0].to_dict()

    return master, all_dfs, latest

@st.cache_data(ttl=1800, show_spinner=False)
def train_models(master_json):
    master=pd.read_json(StringIO(master_json))
    X=master[FEATURE_COLS].values; y_clf=master['Label'].values
    y_reg=master['Fwd_1M'].values*100

    scaler=StandardScaler()
    X_tr,X_te,y_tr_c,y_te_c=train_test_split(X,y_clf,test_size=0.20,random_state=42,stratify=y_clf)
    X_tr_s=scaler.fit_transform(X_tr); X_te_s=scaler.transform(X_te)
    _,_,y_tr_r,y_te_r=train_test_split(X,y_reg,test_size=0.20,random_state=42)

    # XGBoost Classifier
    xgb_clf=xgb.XGBClassifier(n_estimators=250,max_depth=5,learning_rate=0.06,
        subsample=0.80,colsample_bytree=0.80,min_child_weight=5,gamma=0.1,
        reg_alpha=0.1,reg_lambda=1.0,eval_metric='mlogloss',random_state=42,n_jobs=-1)
    xgb_clf.fit(X_tr_s,y_tr_c,eval_set=[(X_te_s,y_te_c)],verbose=False)
    y_pred_c=xgb_clf.predict(X_te_s)
    y_proba_c=xgb_clf.predict_proba(X_te_s)
    acc=float((y_pred_c==y_te_c).mean())

    # CV
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_scores=cross_val_score(xgb_clf,X_tr_s,y_tr_c,cv=cv,scoring='accuracy',n_jobs=-1)

    # ROC
    y_te_bin=label_binarize(y_te_c,classes=[0,1,2])
    macro_auc=float(roc_auc_score(y_te_bin,y_proba_c,multi_class='ovr',average='macro'))

    # RF Regressor
    rf_reg=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=10,
        min_samples_split=20,max_features='sqrt',n_jobs=-1,random_state=42)
    X_tr_rs=scaler.transform(X_tr); X_te_rs=scaler.transform(X_te)
    rf_reg.fit(X_tr_rs,y_tr_r)
    y_pred_r=rf_reg.predict(X_te_rs)
    r2=float(r2_score(y_te_r,y_pred_r))
    rmse=float(np.sqrt(mean_squared_error(y_te_r,y_pred_r)))
    mae=float(mean_absolute_error(y_te_r,y_pred_r))
    ic=float(pd.Series(y_pred_r).corr(pd.Series(y_te_r)))
    dir_acc=float((np.sign(y_pred_r)==np.sign(y_te_r)).mean()*100)

    # Confusion matrix
    cm=confusion_matrix(y_te_c,y_pred_c)
    # Feature importances
    clf_imp=xgb_clf.feature_importances_.tolist()
    reg_imp=rf_reg.feature_importances_.tolist()
    # ROC data
    roc_data=[]
    for i in range(3):
        fpr,tpr,_=roc_curve(y_te_bin[:,i],y_proba_c[:,i])
        auc_v=float(auc(fpr,tpr))
        roc_data.append({'class':i,'fpr':fpr.tolist(),'tpr':tpr.tolist(),'auc':auc_v})
    # Decile analysis
    df_eval=pd.DataFrame({'actual':y_te_r,'predicted':y_pred_r})
    df_eval['decile']=pd.qcut(df_eval['predicted'],10,labels=False,duplicates='drop')
    decile_ret=df_eval.groupby('decile')['actual'].mean().to_dict()

    return {
        'acc':acc,'cv_scores':cv_scores.tolist(),'macro_auc':macro_auc,
        'r2':r2,'rmse':rmse,'mae':mae,'ic':ic,'dir_acc':dir_acc,
        'cm':cm.tolist(),'clf_imp':clf_imp,'reg_imp':reg_imp,
        'roc_data':roc_data,'decile_ret':decile_ret,
        'y_te_r':y_te_r.tolist(),'y_pred_r':y_pred_r.tolist(),
        'y_te_c':y_te_c.tolist(),'y_pred_c':y_pred_c.tolist(),
        'scaler_mean':scaler.mean_.tolist(),'scaler_std':scaler.scale_.tolist(),
        'reg_imp_list':rf_reg.feature_importances_.tolist(),
        'xgb_imp_list':xgb_clf.feature_importances_.tolist(),
    }, rf_reg, scaler


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
      <div style='font-size:17px;font-weight:800;background:linear-gradient(135deg,#00C9FF,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>AlphaLens ML</div>
      <div style='font-size:10px;color:#4A5A80;text-transform:uppercase;letter-spacing:1px;'>Indian Market Intelligence</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    n_stocks = st.slider('Stocks to analyse', 30, len(TICKERS), 60, 10,
        help=f'Nifty LargeMidcap 250 universe — {len(TICKERS)} stocks. 60 recommended for speed.')
    sel_tickers = TICKERS[:n_stocks]

    run_btn = st.button('🚀 Train Models', use_container_width=True)

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#4A5A80;line-height:1.7;'>
    <b style='color:#8B9DC3;'>3 Models:</b><br>
    🤖 XGBoost Classifier<br>
    🌲 Random Forest Regressor<br>
    📐 Portfolio Optimizer<br><br>
    <b style='color:#8B9DC3;'>25 Features:</b><br>
    EMA ratios · RSI · MACD<br>
    CMF · ADX · 52W High · Momentum
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:10px;color:#4A5A80;text-align:center;">Data: Yahoo Finance · Not investment advice</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ml-header">
  <div class="header-title">🤖 AlphaLens ML — Indian Markets</div>
  <div class="header-sub">3 ML Models · XGBoost Classifier · Random Forest Regressor · Portfolio Optimizer · {len(sel_tickers)} Nifty LargeMidcap 250 Stocks · 2Y Data</div>
</div>""", unsafe_allow_html=True)

if 'models_ready' not in st.session_state:
    st.session_state.models_ready = False

if run_btn:
    prog = st.progress(0, text='Downloading market data...')
    with st.spinner(''):
        master, all_dfs, latest = load_and_build(sel_tickers)
    prog.progress(40, text='Training XGBoost & Random Forest...')
    if master is not None and len(master) > 100:
        results, rf_model, scaler = train_models(master.to_json())
        prog.progress(85, text='Building portfolio optimizer...')
        st.session_state.master  = master
        st.session_state.all_dfs = all_dfs
        st.session_state.latest  = latest
        st.session_state.results = results
        st.session_state.rf_model= rf_model
        st.session_state.scaler  = scaler
        st.session_state.models_ready = True
        prog.progress(100, text='Done!')
        st.success(f'✅ Models trained on {len(master):,} samples from {len(all_dfs)} stocks')
    else:
        st.error('Not enough data. Try again or reduce the number of stocks.')

if not st.session_state.models_ready:
    # Landing state
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
      <div style='font-size:48px;margin-bottom:16px;'>🤖</div>
      <div style='font-size:18px;font-weight:700;color:#F0F4FF;margin-bottom:8px;'>Ready to train</div>
      <div style='font-size:13px;color:#8B9DC3;max-width:500px;margin:0 auto;'>
        Select the number of stocks in the sidebar and click <b>Train Models</b>.<br>
        Training takes 2–4 minutes depending on the universe size.
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    cards = [
        ('🤖','XGBoost Classifier','Predicts Buy/Neutral/Sell signal using 25 technical features. Shows ROC curves, confusion matrix, feature importance.','#00C9FF'),
        ('🌲','Random Forest Regressor','Predicts exact 1-month forward return. Decile analysis validates predictive power across return buckets.','#8B5CF6'),
        ('📐','Portfolio Optimizer','Efficient frontier using ML-predicted returns + Ledoit-Wolf covariance. Finds Max Sharpe portfolio, backtests vs Nifty.','#10B981'),
    ]
    for col,(ico,ttl,desc,clr) in zip([c1,c2,c3],cards):
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-top:3px solid {clr};border-radius:12px;padding:20px;text-align:center;'>
              <div style='font-size:32px;margin-bottom:10px;'>{ico}</div>
              <div style='font-size:14px;font-weight:700;color:#F0F4FF;margin-bottom:8px;'>{ttl}</div>
              <div style='font-size:11px;color:#8B9DC3;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Pull from session state ───────────────────────────────────────────────────
master  = st.session_state.master
all_dfs = st.session_state.all_dfs
latest  = st.session_state.latest
R       = st.session_state.results
rf_model= st.session_state.rf_model
scaler  = st.session_state.scaler


# ── Top KPIs ──────────────────────────────────────────────────────────────────
acc_pct  = R['acc']*100
cv_mean  = np.mean(R['cv_scores'])*100
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card" style="--c:#00C9FF;"><div class="kpi-label">Classifier Accuracy</div><div class="kpi-value" style="color:#00C9FF;">{acc_pct:.1f}%</div><div class="kpi-sub">XGBoost test set</div></div>
  <div class="kpi-card" style="--c:#8B5CF6;"><div class="kpi-label">Macro ROC-AUC</div><div class="kpi-value" style="color:#8B5CF6;">{R['macro_auc']:.3f}</div><div class="kpi-sub">One-vs-Rest</div></div>
  <div class="kpi-card" style="--c:#10B981;"><div class="kpi-label">CV Accuracy</div><div class="kpi-value" style="color:#10B981;">{cv_mean:.1f}%</div><div class="kpi-sub">5-Fold mean</div></div>
  <div class="kpi-card" style="--c:#F59E0B;"><div class="kpi-label">R² Score</div><div class="kpi-value" style="color:#F59E0B;">{R['r2']:.3f}</div><div class="kpi-sub">RF Regressor</div></div>
  <div class="kpi-card" style="--c:#F43F5E;"><div class="kpi-label">Info Coefficient</div><div class="kpi-value" style="color:#F43F5E;">{R['ic']:.3f}</div><div class="kpi-sub">Return prediction</div></div>
  <div class="kpi-card" style="--c:#34D399;"><div class="kpi-label">Directional Acc.</div><div class="kpi-value" style="color:#34D399;">{R['dir_acc']:.1f}%</div><div class="kpi-sub">Up/Down correct</div></div>
  <div class="kpi-card" style="--c:#EC4899;"><div class="kpi-label">Training Samples</div><div class="kpi-value" style="color:#EC4899;">{len(master):,}</div><div class="kpi-sub">{master['Ticker'].nunique()} stocks × 1Y</div></div>
  <div class="kpi-card" style="--c:#00C9FF;"><div class="kpi-label">Features</div><div class="kpi-value" style="color:#00C9FF;">25</div><div class="kpi-sub">Technical indicators</div></div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(['📊 Model Overview','🤖 XGBoost Classifier','🌲 RF Regressor','📐 Portfolio Optimizer'])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    l, r = st.columns(2)

    # Label distribution
    with l:
        st.markdown('<div class="sec-title">Label Distribution (1M Forward Return)</div>', unsafe_allow_html=True)
        lbl_counts = master['Label_Name'].value_counts().reindex(['SELL','NEUTRAL','BUY'],fill_value=0)
        fig_lbl = go.Figure(go.Bar(
            x=lbl_counts.index, y=lbl_counts.values,
            marker_color=[C_RED,C_AMBER,C_GREEN],
            text=lbl_counts.values, textposition='outside',
            textfont=dict(color='#F0F4FF', size=12),
        ))
        chl(fig_lbl,'Buy/Neutral/Sell Label Counts',360)
        st.plotly_chart(fig_lbl, use_container_width=True)

    # Feature correlation with forward return
    with r:
        st.markdown('<div class="sec-title">Feature Correlation with 1M Forward Return</div>', unsafe_allow_html=True)
        corr_vals = master[FEATURE_COLS+['Fwd_1M']].corr()['Fwd_1M'].drop('Fwd_1M').sort_values()
        fig_corr = go.Figure(go.Bar(
            x=corr_vals.values,
            y=corr_vals.index,
            orientation='h',
            marker_color=[C_GREEN if v>0 else C_RED for v in corr_vals.values],
            opacity=0.85,
        ))
        chl(fig_corr,'Pearson Correlation with Forward Return',360)
        fig_corr.update_layout(xaxis_title='Correlation', yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Return distribution
    st.markdown('<div class="sec-title">1M Forward Return Distribution</div>', unsafe_allow_html=True)
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(x=master['Fwd_1M']*100, nbinsx=60,
        marker_color=C_CYAN, opacity=0.7, name='All returns'))
    fig_ret.add_vline(x=4,  line_dash='dash', line_color=C_GREEN, annotation_text='BUY threshold (+4%)')
    fig_ret.add_vline(x=-4, line_dash='dash', line_color=C_RED,   annotation_text='SELL threshold (-4%)')
    fig_ret.add_vline(x=master['Fwd_1M'].mean()*100, line_dash='dot', line_color=C_AMBER,
        annotation_text=f'Mean {master["Fwd_1M"].mean()*100:.2f}%')
    chl(fig_ret,'Distribution of 1-Month Forward Returns (%)',300)
    st.plotly_chart(fig_ret, use_container_width=True)

    # Insights
    st.markdown('<div class="sec-title">Model Insights</div>', unsafe_allow_html=True)
    top_clf = sorted(zip(FEATURE_COLS, R['xgb_imp_list']), key=lambda x:-x[1])[:3]
    top_reg = sorted(zip(FEATURE_COLS, R['reg_imp_list']), key=lambda x:-x[1])[:3]
    ins1 = ', '.join([f'<b>{f}</b>' for f,_ in top_clf])
    ins2 = ', '.join([f'<b>{f}</b>' for f,_ in top_reg])
    st.markdown(f"""
    <div class="insight-card">🤖 <b>XGBoost Classifier</b> — Top predictors for Buy/Sell direction: {ins1}</div>
    <div class="insight-card">🌲 <b>Random Forest</b> — Top predictors for return magnitude: {ins2}</div>
    <div class="insight-card">📊 <b>Label balance</b> — BUY: {(master['Label']==2).sum()} | NEUTRAL: {(master['Label']==1).sum()} | SELL: {(master['Label']==0).sum()} samples</div>
    <div class="insight-card">🎯 <b>Directional accuracy {R['dir_acc']:.1f}%</b> means the model correctly predicts whether a stock will rise or fall {R['dir_acc']:.1f}% of the time (50% = random)</div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — XGBOOST CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-title">XGBoost Classifier — Performance</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    cv_arr = np.array(R['cv_scores'])
    for col, val, lbl, sub, clr in [
        (m1, f"{R['acc']*100:.2f}%", 'Test Accuracy', '20% holdout set', C_CYAN),
        (m2, f"{R['macro_auc']:.4f}", 'Macro ROC-AUC', 'One-vs-Rest average', C_PURPLE),
        (m3, f"{cv_arr.mean()*100:.2f}%", 'CV Mean Accuracy', f'±{cv_arr.std()*100:.2f}% std', C_GREEN),
        (m4, '3 classes', 'Task', 'BUY / NEUTRAL / SELL', C_AMBER),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--c:{clr};">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-value" style="color:{clr};">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin:10px 0;"></div>', unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2)

    # Confusion Matrix (heatmap)
    with col_cm:
        st.markdown('<div class="sec-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(R['cm'])
        cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
        labels = ['SELL','NEUTRAL','BUY']
        text_vals = [[f'{cm[i][j]}<br>({cm_norm[i][j]*100:.1f}%)' for j in range(3)] for i in range(3)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm, x=labels, y=labels,
            colorscale=[[0,'rgba(3,9,26,1)'],[0.5,'rgba(0,201,255,0.3)'],[1,'rgba(0,201,255,0.9)']],
            text=text_vals, texttemplate='%{text}',
            textfont=dict(size=12, color='white'),
            showscale=True, zmin=0, zmax=1,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Rate: %{z:.2f}<extra></extra>',
        ))
        chl(fig_cm,'Normalised Confusion Matrix',380)
        fig_cm.update_layout(
            xaxis_title='Predicted', yaxis_title='Actual',
            yaxis=dict(autorange='reversed', color='#8B9DC3'),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curves
    with col_roc:
        st.markdown('<div class="sec-title">ROC Curves — One vs Rest</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        roc_colors = [C_RED, C_AMBER, C_GREEN]
        roc_names  = ['SELL','NEUTRAL','BUY']
        for rd, clr, nm in zip(R['roc_data'], roc_colors, roc_names):
            fig_roc.add_trace(go.Scatter(
                x=rd['fpr'], y=rd['tpr'], mode='lines',
                name=f'{nm} (AUC={rd["auc"]:.3f})',
                line=dict(color=clr, width=2),
                fill='tozeroy', fillcolor=clr.replace('#','rgba(').replace(')',',0.04)') if False else 'none',
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',
            line=dict(color='#4A5A80',dash='dash',width=1)))
        chl(fig_roc,'ROC Curves — XGBoost Classifier',380)
        fig_roc.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02]),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Cross-validation scores
    col_cv, col_fi = st.columns(2)
    with col_cv:
        st.markdown('<div class="sec-title">5-Fold Cross-Validation</div>', unsafe_allow_html=True)
        cv_scores = R['cv_scores']
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(x=[f'Fold {i+1}' for i in range(len(cv_scores))],
            y=[s*100 for s in cv_scores], marker_color=C_CYAN, opacity=0.8,
            text=[f'{s*100:.2f}%' for s in cv_scores], textposition='outside'))
        fig_cv.add_hline(y=np.mean(cv_scores)*100, line_dash='dash', line_color=C_AMBER,
            annotation_text=f'Mean {np.mean(cv_scores)*100:.2f}%', annotation_font_size=10)
        chl(fig_cv,'Cross-Validation Accuracy per Fold',320)
        fig_cv.update_layout(yaxis_range=[0,100])
        st.plotly_chart(fig_cv, use_container_width=True)

    # Feature importance
    with col_fi:
        st.markdown('<div class="sec-title">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        fi = sorted(zip(FEATURE_COLS, R['xgb_imp_list']), key=lambda x:x[1])
        fig_fi = go.Figure(go.Bar(
            y=[f[0] for f in fi], x=[f[1] for f in fi], orientation='h',
            marker_color=[C_CYAN if v >= np.percentile([x[1] for x in fi],75) else C_PURPLE
                          for _,v in fi],
            opacity=0.85,
        ))
        chl(fig_fi,'Feature Importance (Gain)',320)
        fig_fi.update_layout(yaxis=dict(tickfont=dict(size=8)), xaxis_title='Importance Score')
        st.plotly_chart(fig_fi, use_container_width=True)

    # Predict on individual stock
    st.markdown('<div class="sec-title">🔮 Predict Signal for a Specific Stock</div>', unsafe_allow_html=True)
    avail = list(latest.keys())
    if avail:
        pred_stock = st.selectbox('Select stock', avail,
            format_func=lambda x: x.replace('.NS',''), key='clf_pred_stock')
        if pred_stock in latest:
            feat_vec = np.array([latest[pred_stock].get(f, 0) for f in FEATURE_COLS]).reshape(1,-1)
            feat_scaled = scaler.transform(feat_vec)
            # rebuild scaler from saved params
            sc2 = StandardScaler(); sc2.mean_=np.array(R['scaler_mean']); sc2.scale_=np.array(R['scaler_std'])
            feat_scaled2 = sc2.transform(feat_vec)

            xgb_pred_cls = None
            try:
                xgb_pred_cls = 1  # placeholder; use rf_model approach below
            except: pass

            pred_proba = rf_model.predict([feat_vec[0]])[0]
            lbl_map = {0:'SELL',1:'NEUTRAL',2:'BUY'}
            p1,p2,p3 = st.columns(3)
            proba_colors = [C_RED, C_AMBER, C_GREEN]
            proba_labels = ['SELL','NEUTRAL','BUY']
            # Use feature vector with RF as proxy for class probabilities
            # Show predicted return instead
            pred_ret = rf_model.predict(feat_vec)[0]
            pred_class = 'BUY' if pred_ret > 4 else ('SELL' if pred_ret < -4 else 'NEUTRAL')
            clr_map = {'BUY':C_GREEN,'SELL':C_RED,'NEUTRAL':C_AMBER}
            badge_map = {'BUY':'badge-green','SELL':'badge-red','NEUTRAL':'badge-amber'}
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px 20px;'>
              <div style='font-size:12px;color:#8B9DC3;margin-bottom:6px;'>Predicted for <b style='color:#F0F4FF;'>{pred_stock.replace(".NS","")}</b></div>
              <div style='display:flex;gap:20px;align-items:center;'>
                <div>
                  <div style='font-size:10px;color:#4A5A80;text-transform:uppercase;'>Predicted 1M Return</div>
                  <div style='font-size:28px;font-weight:800;font-family:"JetBrains Mono",monospace;color:{clr_map[pred_class]};'>{pred_ret:+.2f}%</div>
                </div>
                <div>
                  <div style='font-size:10px;color:#4A5A80;text-transform:uppercase;'>Signal</div>
                  <div style='margin-top:4px;'><span class='badge {badge_map[pred_class]}'>{pred_class}</span></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RF REGRESSOR
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-title">Random Forest Regressor — 1M Return Prediction</div>', unsafe_allow_html=True)
    rm1,rm2,rm3,rm4,rm5 = st.columns(5)
    for col, val, lbl, clr in [
        (rm1,f"{R['r2']:.4f}",'R² Score',C_CYAN),
        (rm2,f"{R['rmse']:.2f}%",'RMSE',C_PURPLE),
        (rm3,f"{R['mae']:.2f}%",'MAE',C_GREEN),
        (rm4,f"{R['ic']:.4f}",'Info Coefficient',C_AMBER),
        (rm5,f"{R['dir_acc']:.1f}%",'Directional Accuracy',C_RED),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--c:{clr};">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-value" style="color:{clr};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin:10px 0;"></div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)

    y_te_r   = np.array(R['y_te_r'])
    y_pred_r = np.array(R['y_pred_r'])

    # Actual vs Predicted scatter
    with sc1:
        st.markdown('<div class="sec-title">Actual vs Predicted Returns</div>', unsafe_allow_html=True)
        lim = max(abs(y_te_r).max(), abs(y_pred_r).max()) * 1.1
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_te_r, y=y_pred_r, mode='markers',
            marker=dict(color=C_CYAN, size=4, opacity=0.4),
            name='Predictions',
            hovertemplate='Actual: %{x:.2f}%<br>Predicted: %{y:.2f}%<extra></extra>',
        ))
        fig_sc.add_trace(go.Scatter(x=[-lim,lim],y=[-lim,lim],mode='lines',
            name='Perfect',line=dict(color=C_RED,dash='dash',width=1.5)))
        chl(fig_sc,f'Actual vs Predicted — R²={R["r2"]:.3f}',380)
        fig_sc.update_layout(xaxis_title='Actual 1M Return (%)',yaxis_title='Predicted 1M Return (%)',
            xaxis=dict(range=[-lim,lim],zeroline=True,zerolinecolor='rgba(255,255,255,0.1)'),
            yaxis=dict(range=[-lim,lim],zeroline=True,zerolinecolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Residuals
    with sc2:
        st.markdown('<div class="sec-title">Residual Distribution</div>', unsafe_allow_html=True)
        residuals = y_te_r - y_pred_r
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=residuals,nbinsx=60,marker_color=C_PURPLE,opacity=0.75,name='Residuals'))
        fig_res.add_vline(x=0,line_dash='dash',line_color=C_AMBER,annotation_text='Zero error')
        fig_res.add_vline(x=residuals.mean(),line_dash='dot',line_color=C_RED,
            annotation_text=f'Mean={residuals.mean():.2f}%',annotation_font_size=9)
        chl(fig_res,f'Residuals — Std={residuals.std():.2f}%',380)
        fig_res.update_layout(xaxis_title='Residual (%)',yaxis_title='Frequency')
        st.plotly_chart(fig_res, use_container_width=True)

    # Decile analysis
    st.markdown('<div class="sec-title">Decile Analysis — Predictive Power Validation</div>', unsafe_allow_html=True)
    decile_ret = R['decile_ret']
    dec_vals = [decile_ret.get(str(k), decile_ret.get(k,0)) for k in range(10)]
    fig_dec = go.Figure(go.Bar(
        x=[f'D{i+1}' for i in range(len(dec_vals))],
        y=dec_vals,
        marker_color=[C_GREEN if v>0 else C_RED for v in dec_vals],
        text=[f'{v:.2f}%' for v in dec_vals],textposition='outside',
        opacity=0.85,
    ))
    fig_dec.add_hline(y=np.mean(dec_vals),line_dash='dash',line_color=C_AMBER,
        annotation_text=f'Mean {np.mean(dec_vals):.2f}%',annotation_font_size=10)
    fig_dec.add_hline(y=0,line_color='rgba(255,255,255,0.1)',line_width=1)
    chl(fig_dec,'Average Actual Return by Predicted Return Decile (D1=Lowest Predicted → D10=Highest)',320)
    fig_dec.update_layout(xaxis_title='Predicted Return Decile',yaxis_title='Avg Actual Return (%)',
        yaxis=dict(zeroline=True,zerolinecolor='rgba(255,255,255,0.15)'))
    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    <b>How to read this chart:</b> Stocks in the highest predicted-return decile (D10) should consistently deliver higher actual returns than D1.
    A monotonically increasing pattern from D1→D10 confirms the model has real predictive power beyond just memorising training data.
    </div>""", unsafe_allow_html=True)

    # RF Feature importance
    col_fi2, col_pred2 = st.columns(2)
    with col_fi2:
        st.markdown('<div class="sec-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fi2 = sorted(zip(FEATURE_COLS, R['reg_imp_list']), key=lambda x:x[1])[-15:]
        fig_fi2 = go.Figure(go.Bar(
            y=[f[0] for f in fi2], x=[f[1] for f in fi2], orientation='h',
            marker_color=[C_GREEN if v>=np.percentile([x[1] for x in fi2],75) else C_CYAN for _,v in fi2],
            opacity=0.85,
        ))
        chl(fig_fi2,'Top 15 Features',380)
        fig_fi2.update_layout(yaxis=dict(tickfont=dict(size=9)),xaxis_title='Importance Score')
        st.plotly_chart(fig_fi2, use_container_width=True)

    with col_pred2:
        st.markdown('<div class="sec-title">📈 Return Predictions — All Stocks</div>', unsafe_allow_html=True)
        if latest:
            preds = {}
            for tkr, feats in latest.items():
                fv = np.array([feats.get(f,0) for f in FEATURE_COLS]).reshape(1,-1)
                try:
                    p = rf_model.predict(fv)[0]
                    preds[tkr] = round(float(p),2)
                except: pass
            pred_df = pd.DataFrame({'Ticker':list(preds.keys()),
                                    'Predicted_1M_Pct':list(preds.values())})
            pred_df['Name'] = pred_df['Ticker'].str.replace('.NS','')
            pred_df = pred_df.sort_values('Predicted_1M_Pct',ascending=True)
            fig_pred = go.Figure(go.Bar(
                y=pred_df['Name'], x=pred_df['Predicted_1M_Pct'], orientation='h',
                marker_color=[C_GREEN if v>0 else C_RED for v in pred_df['Predicted_1M_Pct']],
                text=[f'{v:+.1f}%' for v in pred_df['Predicted_1M_Pct']],
                textposition='outside', textfont=dict(size=9),
                opacity=0.85,
            ))
            fig_pred.add_vline(x=0,line_color='rgba(255,255,255,0.15)',line_width=1)
            chl(fig_pred,'Predicted 1M Return per Stock',max(380, len(pred_df)*16))
            fig_pred.update_layout(yaxis=dict(tickfont=dict(size=9)),
                xaxis_title='Predicted 1M Return (%)',
                xaxis=dict(zeroline=True,zerolinecolor='rgba(255,255,255,0.15)'))
            st.plotly_chart(fig_pred, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-title">ML-Enhanced Portfolio Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Uses <b>Model 2's predicted returns</b> as expected returns + <b>Ledoit-Wolf covariance shrinkage</b>
    to find the optimal portfolio on the efficient frontier. Monte Carlo simulation of 5,000 random portfolios.
    </div>""", unsafe_allow_html=True)

    # Get predicted returns for latest data
    if latest:
        preds_opt = {}
        for tkr, feats in latest.items():
            fv = np.array([feats.get(f,0) for f in FEATURE_COLS]).reshape(1,-1)
            try:
                p = rf_model.predict(fv)[0]
                preds_opt[tkr] = float(p) / 100  # as decimal monthly
            except: pass

        # Download price series
        END = datetime.today(); START = END - timedelta(days=365)
        price_data = {}
        with st.spinner('Building covariance matrix...'):
            for tkr in list(preds_opt.keys()):
                try:
                    raw = yf.download(tkr, start=START, end=END, auto_adjust=True, progress=False)
                    if isinstance(raw.columns, pd.MultiIndex): raw = raw.droplevel(1, axis=1)
                    if len(raw) >= 100: price_data[tkr] = raw['Close']
                except: pass

        price_df   = pd.DataFrame(price_data).dropna(axis=1, how='any')
        returns_df = price_df.pct_change().dropna()
        common     = [t for t in preds_opt if t in returns_df.columns]

        if len(common) >= 5:
            returns_df = returns_df[common]
            mu_monthly = np.array([preds_opt[t] for t in common])
            mu_annual  = mu_monthly * 12

            # Ledoit-Wolf covariance
            lw = LedoitWolf()
            lw.fit(returns_df.values)
            cov_annual = lw.covariance_ * 252

            n = len(common)
            RF_RATE = 0.065

            # Monte Carlo
            N = 4000
            np.random.seed(42)
            p_ret=[]; p_vol=[]; p_sr=[]; p_wts=[]
            for _ in range(N):
                w = np.random.dirichlet(np.ones(n))
                r = float(np.dot(w, mu_annual))
                v = float(np.sqrt(w @ cov_annual @ w))
                s = (r - RF_RATE) / v if v > 0 else 0
                p_ret.append(r); p_vol.append(v); p_sr.append(s); p_wts.append(w)

            p_ret=np.array(p_ret); p_vol=np.array(p_vol)
            p_sr=np.array(p_sr);   p_wts=np.array(p_wts)

            max_sr_idx = p_sr.argmax();   min_v_idx = p_vol.argmin()
            ms_wts = p_wts[max_sr_idx];   mv_wts = p_wts[min_v_idx]
            ms_ret=p_ret[max_sr_idx]; ms_vol=p_vol[max_sr_idx]; ms_sr=p_sr[max_sr_idx]
            mv_ret=p_ret[min_v_idx];  mv_vol=p_vol[min_v_idx]

            # KPIs
            po1,po2,po3,po4 = st.columns(4)
            for col,val,lbl,clr in [
                (po1,f'{ms_ret*100:.1f}%','Max Sharpe Return',C_GREEN),
                (po2,f'{ms_vol*100:.1f}%','Max Sharpe Volatility',C_AMBER),
                (po3,f'{ms_sr:.3f}','Max Sharpe Ratio',C_CYAN),
                (po4,f'{lw.shrinkage_:.4f}','LW Shrinkage α',C_PURPLE),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card" style="--c:{clr};">
                      <div class="kpi-label">{lbl}</div>
                      <div class="kpi-value" style="color:{clr};">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div style="margin:10px 0;"></div>', unsafe_allow_html=True)
            ef_l, ef_r = st.columns(2)

            # Efficient Frontier
            with ef_l:
                st.markdown('<div class="sec-title">Efficient Frontier</div>', unsafe_allow_html=True)
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=p_vol*100, y=p_ret*100, mode='markers',
                    marker=dict(color=p_sr, colorscale='viridis', size=3, opacity=0.4,
                                colorbar=dict(title='Sharpe', thickness=10, len=0.7)),
                    name='Portfolios',
                    hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>',
                ))
                fig_ef.add_trace(go.Scatter(x=[ms_vol*100],y=[ms_ret*100],mode='markers',
                    marker=dict(color=C_AMBER,size=16,symbol='star'),name=f'Max Sharpe (SR={ms_sr:.2f})'))
                fig_ef.add_trace(go.Scatter(x=[mv_vol*100],y=[mv_ret*100],mode='markers',
                    marker=dict(color=C_GREEN,size=12,symbol='diamond'),name='Min Volatility'))
                x_cml=np.linspace(0,p_vol.max()*100*1.1,80)
                y_cml=RF_RATE*100 + ms_sr*x_cml
                fig_ef.add_trace(go.Scatter(x=x_cml,y=y_cml,mode='lines',name='Capital Market Line',
                    line=dict(color=C_RED,dash='dash',width=1.5)))
                fig_ef.add_hline(y=RF_RATE*100,line_dash='dot',line_color='#4A5A80',
                    annotation_text=f'Risk-Free {RF_RATE*100:.1f}%',annotation_font_size=9)
                chl(fig_ef,'Efficient Frontier (ML-Enhanced)',440)
                fig_ef.update_layout(xaxis_title='Annual Volatility (%)',yaxis_title='Expected Annual Return (%)')
                st.plotly_chart(fig_ef, use_container_width=True)

            # Portfolio weights
            with ef_r:
                st.markdown('<div class="sec-title">Max Sharpe Portfolio — Weights</div>', unsafe_allow_html=True)
                wt_series = pd.Series(ms_wts, index=[t.replace('.NS','') for t in common])
                top_wts   = wt_series.nlargest(15).sort_values()
                wt_colors = [C_CYAN,C_PURPLE,C_GREEN,C_AMBER,C_RED]*3
                fig_wts = go.Figure(go.Bar(
                    y=top_wts.index, x=top_wts.values*100, orientation='h',
                    marker_color=wt_colors[:len(top_wts)],
                    text=[f'{v*100:.1f}%' for v in top_wts.values],
                    textposition='outside', textfont=dict(size=10),
                    opacity=0.85,
                ))
                chl(fig_wts,'Top 15 Holdings (%)',440)
                fig_wts.update_layout(xaxis_title='Portfolio Weight (%)',
                    yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig_wts, use_container_width=True)

            # Backtest vs Nifty
            st.markdown('<div class="sec-title">Portfolio Backtest vs Nifty 50</div>', unsafe_allow_html=True)
            with st.spinner('Running backtest...'):
                try:
                    nifty = yf.download('^NSEI',start=START,end=END,auto_adjust=True,progress=False)
                    if isinstance(nifty.columns,pd.MultiIndex): nifty=nifty.droplevel(1,axis=1)
                    nifty_ret_s = nifty['Close'].pct_change().dropna()

                    TOP_N = min(12, len(common))
                    top_tkrs = pd.Series(ms_wts,index=common).nlargest(TOP_N)
                    top_tkrs = top_tkrs / top_tkrs.sum()
                    port_r   = returns_df[top_tkrs.index].dot(top_tkrs.values)
                    port_r   = port_r[port_r.index.isin(nifty_ret_s.index)]
                    ni_r     = nifty_ret_s.reindex(port_r.index)

                    cum_p = (1+port_r).cumprod()
                    cum_n = (1+ni_r).cumprod()
                    tp = (cum_p.iloc[-1]-1)*100; tn = (cum_n.iloc[-1]-1)*100
                    sp = ((port_r.mean()*252 - RF_RATE)/(port_r.std()*np.sqrt(252)))

                    bk1,bk2,bk3,bk4 = st.columns(4)
                    for col,val,lbl,clr in [
                        (bk1,f'{tp:+.2f}%','ML Portfolio 1Y',C_CYAN),
                        (bk2,f'{tn:+.2f}%','Nifty 50 1Y',C_AMBER),
                        (bk3,f'{tp-tn:+.2f}%','Excess Return',C_GREEN if tp>tn else C_RED),
                        (bk4,f'{sp:.3f}','Portfolio Sharpe',C_PURPLE),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="kpi-card" style="--c:{clr};">
                              <div class="kpi-label">{lbl}</div>
                              <div class="kpi-value" style="color:{clr};">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown('<div style="margin:8px 0;"></div>', unsafe_allow_html=True)
                    fig_bt = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.70,0.30],vertical_spacing=0.05)
                    fig_bt.add_trace(go.Scatter(x=cum_p.index,y=cum_p.values,
                        name='ML Portfolio',line=dict(color=C_CYAN,width=2.5)),row=1,col=1)
                    fig_bt.add_trace(go.Scatter(x=cum_n.index,y=cum_n.values,
                        name='Nifty 50',line=dict(color=C_AMBER,width=2,dash='dash')),row=1,col=1)
                    excess=(cum_p/cum_n-1)*100
                    fill_green=go.Scatter(x=excess.index,y=np.where(excess.values>=0,excess.values,0),
                        fill='tozeroy',fillcolor='rgba(16,185,129,0.15)',line_color='rgba(0,0,0,0)',
                        name='Outperform',showlegend=False)
                    fill_red=go.Scatter(x=excess.index,y=np.where(excess.values<0,excess.values,0),
                        fill='tozeroy',fillcolor='rgba(244,63,94,0.15)',line_color='rgba(0,0,0,0)',
                        name='Underperform',showlegend=False)
                    fig_bt.add_trace(fill_green,row=2,col=1)
                    fig_bt.add_trace(fill_red,row=2,col=1)
                    fig_bt.add_hline(y=0,line_color='rgba(255,255,255,0.1)',row=2,col=1)
                    fig_bt.update_layout(
                        title=dict(text=f'ML Portfolio vs Nifty 50 (1Y) — Portfolio: {tp:+.1f}% | Nifty: {tn:+.1f}% | α: {tp-tn:+.1f}%',
                                   font=dict(size=12,color='#8B9DC3')),
                        paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                        font=dict(family='Inter',color='#8B9DC3',size=11),
                        height=480,margin=dict(l=40,r=20,t=40,b=40),
                        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=10)),
                        hovermode='x unified',
                    )
                    for ax in ['xaxis','xaxis2']:
                        fig_bt.update_layout(**{ax:dict(showgrid=False,color='#4A5A80')})
                    for ax in ['yaxis','yaxis2']:
                        fig_bt.update_layout(**{ax:dict(showgrid=True,gridcolor=CHART_GRID,color='#4A5A80')})
                    st.plotly_chart(fig_bt, use_container_width=True)
                except Exception as e:
                    st.warning(f'Backtest error: {e}')
        else:
            st.warning('Not enough common tickers for portfolio optimization. Try loading more stocks.')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-txt">
  AlphaLens ML · 3 Models: XGBoost · Random Forest · Portfolio Optimizer<br>
  Data: Yahoo Finance · Not investment advice · Educational purposes only
</div>""", unsafe_allow_html=True)
