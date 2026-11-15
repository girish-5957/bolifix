"""BOLIFIX Showcase — public-safe demonstration frontend.

Uses direct Gnani Prisma/Timbre requests only after explicit user actions.
No database, uploads directory, saved audio, or training labels are created.
The existing frontend/app.py remains the full local research cockpit.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from backend.clarification import suggest_clarification
from backend.dual_asr import compare_hypotheses
from backend.dual_recovery import next_dual_recovery_step
from backend.recovery_flow import next_recovery_step
from backend.tts import clarification_question
from backend.showcase_workflow import make_receipt, receipt_json, run_local_whisper
from backend.learning_lifecycle import session_learning_status, LEARNING_STAGES

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
STT_URL = "https://api.vachana.ai/stt/v3"
MAX_UPLOAD = 10 * 1024 * 1024
EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}

st.set_page_config(
    page_title="BOLIFIX — Voice understanding, recovered",
    page_icon="◉", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""<style>
/* BOLIFIX / Product interface. Native Streamlit controls remain real and accessible. */
:root {--ink:#163633;--muted:#56726d;--brand:#0a7766;--brand-dark:#075e54;
       --paper:#fafbf8;--line:#dce9e1;--light:#f1f8f3;}
html,body,[class*="css"]{font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;}
.stApp,[data-testid="stAppViewContainer"]{background:var(--paper)!important;color:var(--ink)!important;}
.block-container{max-width:1180px!important;padding-top:.75rem!important;padding-bottom:2.8rem!important;}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
[data-testid="stSidebar"]{background:#f3f8f5;}
.stMarkdown, [data-testid="stMarkdownContainer"]{color:var(--ink);}
[data-testid="stMarkdownContainer"] p{font-size:1.07rem;line-height:1.7;}
h1,h2,h3{letter-spacing:-.035em!important;color:var(--ink);}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:1.5rem;border-bottom:1px solid #dce6e0;}
[data-testid="stTabs"] button{font-size:1.01rem!important;padding:.9rem .1rem!important;font-weight:670!important;color:#53706b!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--brand)!important;}
[data-testid="stButton"] button{border-radius:12px;font-weight:680;min-height:2.8rem;transition:background .14s,border-color .14s;}
[data-testid="stButton"] button[kind="primary"]{background:var(--brand);border-color:var(--brand);color:white;min-height:3.25rem;font-size:1.03rem;}
[data-testid="stButton"] button[kind="primary"]:hover{background:var(--brand-dark);border-color:var(--brand-dark);color:white;}
[data-testid="stButton"] button:not([kind="primary"]):hover{border-color:var(--brand);color:var(--brand);}
[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]{border-radius:17px!important;}
[data-testid="stExpander"]{background:#fff;border-color:var(--line);}
[data-testid="stAudioInput"],[data-testid="stFileUploader"]{border-radius:15px;background:#f0f8f4;border:1px solid #d5e9df;padding:16px!important;}
[data-testid="stAudioInput"]{min-height:92px;}
[data-testid="stFileUploader"] section{border:0!important;padding:8px!important;}
[data-testid="stSelectbox"] [data-baseweb="select"]>div{border-radius:10px!important;border-color:#d4e2da!important;background:#fbfdfb!important;}
[data-testid="stCheckbox"] label{font-size:.98rem;}
[data-testid="stAlert"]{border-radius:12px;}
/* Slim brand bar */
.bf-brand{display:flex;justify-content:space-between;align-items:center;gap:12px;border-bottom:1px solid #e0eae3;padding:7px 0 16px;margin-bottom:19px;}
.bf-brand-name{display:flex;align-items:center;gap:10px;font-size:1.32rem;font-weight:860;letter-spacing:-.046em;color:#193a35;}
.bf-brand-glyph{display:inline-flex;justify-content:center;align-items:center;width:35px;height:35px;background:#0a7766;color:#fff;border-radius:11px;font-size:1rem;letter-spacing:0;}
.bf-brand-right{font-size:.81rem;color:#668079;font-weight:590;}
/* Clean hero: one claim, immediately readable; no simulated transcript */
.bf-hero-clear{position:relative;overflow:hidden;padding:39px 43px 32px;background:linear-gradient(112deg,#eff8ef 0%,#f6fbf6 61%,#e5f4ed 100%);
 border:1px solid #d7e9de;border-radius:23px;}
.bf-hero-clear:before{content:"";position:absolute;width:390px;height:390px;right:-150px;top:-155px;border:1px solid #8ccfba5c;
 border-radius:50%;box-shadow:0 0 0 49px #2a95700a,0 0 0 108px #2a957008;pointer-events:none;}
.bf-hero-clear>*{position:relative;z-index:1;}
.bf-overline{font-size:.78rem;color:#127666;letter-spacing:.13em;font-weight:790;text-transform:uppercase;margin:0 0 17px;}
.bf-hero-clear h1{font-size:clamp(2.55rem,4.05vw,4.12rem)!important;line-height:1.075!important;font-weight:840!important;
 max-width:830px;color:#133833!important;margin:0!important;letter-spacing:-.063em!important;}
.bf-hero-clear h1 span{color:#09846e;}
.bf-hero-sub{font-size:1.19rem;line-height:1.67;color:#48685f;max-width:840px;margin:19px 0 24px;}
.bf-hero-actions{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:0;}
.bf-hero-cta{display:inline-flex;align-items:center;gap:12px;text-decoration:none!important;background:#0a7766;border:1px solid #0a7766;
 padding:13px 20px;color:white!important;border-radius:12px;font-size:.98rem;font-weight:760;box-shadow:0 5px 16px #0c79651b;}
.bf-hero-cta:hover{background:#065c50;}
.bf-hero-cta-caption{font-size:.87rem;color:#58736d;}
/* Two-stage architecture, only essential stages above the fold */
.bf-process{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin:16px 0 21px;}
.bf-process-card{padding:22px 24px;background:#fff;border:1px solid #dfeae3;border-radius:17px;min-width:0;box-shadow:0 4px 17px #254b3110;}
.bf-process-card.future{background:#f6f9f6;}
.bf-process-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;}
.bf-process-index{color:#0b7869;font-size:.78rem;letter-spacing:.10em;font-weight:780;text-transform:uppercase;}
.bf-status{display:inline-block;background:#e6f5e9;color:#176a4c;border:1px solid #c9e7d2;border-radius:999px;
 font-size:.73rem;font-weight:750;padding:5px 10px;white-space:nowrap;}
.bf-status.future{background:#eef2ee;border-color:#dce5dd;color:#62756c;}
.bf-process-card h2{font-size:1.5rem!important;line-height:1.25;margin:0 0 7px!important;letter-spacing:-.045em;}
.bf-process-card p{font-size:1.055rem!important;line-height:1.62!important;color:#58736a!important;margin:0!important;}
.bf-process-steps{font-size:.91rem;font-weight:680;color:#0b7564;margin-top:13px;line-height:1.55;}
.bf-process-note{font-size:.87rem;color:#5c766b;margin-top:8px;}
/* Session bar is deliberately quiet */
.bf-session-indicator{background:#f1f8f3;border:1px solid #e0ede4;padding:10px 15px;border-radius:12px;
 display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:8px 0 15px;}
.bf-session-label{font-size:.77rem;color:#0b7462;font-weight:790;letter-spacing:.05em;}
.bf-session-detail{font-size:.86rem;color:#49685e;}
.bf-pill{display:inline-flex;align-items:center;gap:7px;font-size:.81rem;color:#307363;font-weight:670;white-space:nowrap;}
.bf-pill-dot{width:7px;height:7px;background:#27a077;border-radius:100%;}
/* Voice studio; native form is the product, not secondary copy */
.bf-studio-heading{display:flex;justify-content:space-between;gap:18px;align-items:center;margin:23px 0 13px;}
.bf-section-head{font-size:2.04rem;font-weight:830;letter-spacing:-.049em;line-height:1.15;color:#163a35;}
.bf-studio-subtitle{font-size:1.12rem;color:#4c6e62;margin-top:8px;line-height:1.6;max-width:760px;}
.bf-studio-badge{background:#e7f4eb;color:#176c55;border-radius:999px;padding:7px 11px;font-size:.77rem;font-weight:720;white-space:nowrap;}
.bf-journey{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:4px 0 14px;}
.bf-journey-cell{display:flex;align-items:center;gap:9px;min-width:0;background:#fff;border:1px solid #e1ebe4;border-radius:12px;padding:12px 11px;}
.bf-journey-cell.bf-current{background:#ebf8f0;border-color:#0e9076;box-shadow:inset 0 0 0 1px #0e9076;}
.bf-journey-cell.bf-done{background:#eef9f0;border-color:#a3d1b9;}
.bf-num{display:flex;align-items:center;justify-content:center;flex-shrink:0;width:31px;height:31px;border-radius:10px;background:#edf2ed;color:#527467;font-weight:770;font-size:.78rem;}
.bf-current .bf-num{background:#0d7869;color:white;}.bf-done .bf-num{background:#166e56;color:white;}
.bf-journey-label{font-size:.88rem;font-weight:800;color:#254b41;}
.bf-journey-description{font-size:.75rem;color:#759087;margin-top:2px;}
.bf-studio-kicker{color:#13816b;font-weight:780;letter-spacing:.11em;font-size:.73rem;text-transform:uppercase;margin:0 0 8px;}
.bf-input-lead{font-size:1.4rem;font-weight:800;letter-spacing:-.028em;margin:2px 0 4px;color:#20443e;}
.bf-input-help{font-size:.93rem;color:#637d72;margin-bottom:13px;line-height:1.5;}
.bf-privacy-note{font-size:.84rem;color:#6b8178;line-height:1.55;margin:8px 0 12px;}
.bf-step{margin:24px 0 9px;color:#0b7c69;font-weight:780;font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;}
.bf-transcript{background:#eff8f1;border-left:4px solid #08826e;border-radius:12px;padding:22px 24px;font-size:1.25rem;line-height:1.7;color:#1a3934;overflow-wrap:anywhere;}
.bf-result{background:#103f37;color:white;border-radius:17px;padding:27px 29px;margin:14px 0;}
.bf-result-eyebrow{color:#a9f1cf;text-transform:uppercase;letter-spacing:.12em;font-size:.78rem;font-weight:800;}
.bf-result-value{color:white;font-size:clamp(2rem,3.2vw,3.2rem);font-weight:840;line-height:1.2;margin:10px 0;overflow-wrap:anywhere;}
.bf-result-note{color:#cde8dd;font-size:.92rem;}
/* Research and advanced sections: organized, no homepage walls of cards */
.bf-line{border-top:1px solid #e0eae3;margin:22px 0;}
.bf-muted{font-size:1rem;color:#60766e;line-height:1.65;}
.bf-learning-banner{background:#edf7ef;border:1px solid #d3e8da;border-radius:16px;padding:23px 27px;margin:13px 0 24px;}
.bf-learning-eyebrow{color:#0c7e67;font-size:.72rem;font-weight:800;letter-spacing:.12em;}
.bf-learning-headline{color:#1d4b3d;font-weight:790;font-size:1.48rem;letter-spacing:-.028em;margin:6px 0;}
.bf-learning-banner p{font-size:.98rem;color:#506e62;line-height:1.62;margin:0;}
.bf-learning-stage{display:flex;gap:12px;background:#fff;border:1px solid #dceae0;border-radius:12px;padding:14px;margin:9px 0;align-items:flex-start;}
.bf-learn-complete{background:#e9f8ef;border-color:#a9d6bb;}
.bf-learning-icon{display:flex;align-items:center;justify-content:center;flex-shrink:0;width:31px;height:31px;
 border-radius:50%;background:#eff4ef;color:#4b7964;font-weight:800;}
.bf-learn-complete .bf-learning-icon{background:#0b806d;color:white;}
.bf-learning-stage strong{font-size:.99rem;}.bf-learning-desc{font-size:.87rem;line-height:1.56;color:#667d71;margin-top:3px;}
.bf-learning-future{border-bottom:1px solid #dfebe2;padding:13px 0;}
.bf-learning-future strong{font-size:.98rem;color:#204a3d;}
.bf-learning-tag{font-size:.72rem;color:#63786f;margin-left:10px;}
.bf-learning-explainer{background:#f2f7f2;border:1px solid #e1eae2;border-radius:11px;padding:19px 21px;color:#46695a;font-size:.99rem;line-height:1.62;}
.bf-edu-card{background:#fff;border:1px solid #dde9df;border-radius:15px;padding:20px 22px;margin:8px 0 14px;}
.bf-edu-kicker{font-size:.73rem;letter-spacing:.12em;font-weight:800;color:#147963;}
.bf-edu-card h3{font-size:1.2rem;line-height:1.33;margin:8px 0;color:#194239;}
.bf-edu-card p{font-size:.95rem;line-height:1.66;color:#5a766b;margin:0;}
.bf-system-panel{background:#173f39;border-radius:17px;padding:23px;color:#fff;}
.bf-system-panel h3{color:#fff;margin:7px 0;}.bf-system-panel p{color:#cee4d8;line-height:1.65;}
.bf-system-row{display:flex;gap:11px;border-top:1px solid #49766a;padding:15px 0;color:#fff;}
.bf-system-row strong{display:block;font-size:.99rem;color:#fff;}
.bf-system-row span{display:block;font-size:.86rem;color:#d2e6dd;line-height:1.5;}
.bf-system-dot{display:flex;flex-shrink:0;align-items:center;justify-content:center;width:32px;height:32px;
 color:#ddfff0;background:#286458;border-radius:9px;font-weight:750;}
.bf-narrative-header{margin:19px 0;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.bf-narrative-title{font-size:1.28rem;font-weight:800;}.bf-narrative-detail{color:#668176;font-size:.84rem;}
.bf-metric{border:1px solid #dce8de;border-radius:12px;padding:17px;background:#fff;min-height:105px;}
.bf-metric b{display:block;font-size:1.08rem;color:#174339;margin-bottom:6px;}.bf-metric small{color:#698176;line-height:1.45;}
.bf-foot{font-size:.85rem;color:#698076;line-height:1.6;}
@media(max-width:800px){.bf-process{grid-template-columns:1fr;}.bf-hero-clear{padding:26px 23px;}
 .bf-hero-clear h1{font-size:2.6rem!important;}.bf-journey{grid-template-columns:repeat(2,minmax(0,1fr));}
 .bf-brand-right{display:none;}.bf-studio-heading{align-items:flex-start;flex-direction:column;}}
@media(max-width:450px){.bf-journey{gap:5px;}.bf-journey-cell{padding:9px 7px;}.bf-journey-description{display:none;}
 .bf-process-card{padding:17px;}.bf-hero-clear h1{font-size:2.32rem!important;}}
@media(prefers-reduced-motion:reduce){[data-testid="stButton"] button{transition:none;}}

/* FIRST-VISIT STORY: one reason, one example, one clear product journey. */
.bf-hero-clear{padding:31px 35px!important;background:linear-gradient(126deg,#f3faf6,#f9fcf8 65%,#e9f7ef)!important;}
.bf-hero-clear:before{right:-260px;top:-250px;opacity:.55;}
.bf-intro-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(310px,.8fr);gap:32px;align-items:start;}
.bf-intro-copy{min-width:0;}
.bf-hero-clear h1{font-size:clamp(2.35rem,3.5vw,3.6rem)!important;max-width:750px!important;line-height:1.09!important;margin:6px 0 0!important;}
.bf-hero-sub{font-size:1.13rem!important;line-height:1.57!important;max-width:690px!important;margin:18px 0 20px!important;color:#3c6257!important;}
.bf-hero-cta{font-size:1.02rem!important;padding:14px 20px!important;}
.bf-scenario{border:1px solid #d9e7df;border-radius:20px;background:#fff;padding:22px 23px;box-shadow:0 10px 33px #17442b12;position:relative;min-width:0;}
.bf-scenario-kicker{color:#1a7a65;letter-spacing:.105em;font-size:.74rem;font-weight:820;text-transform:uppercase;}
.bf-scenario h2{font-size:1.32rem!important;line-height:1.31;margin:8px 0 11px!important;letter-spacing:-.028em!important;}
.bf-scenario-quote{background:#f1f8f4;border-radius:11px;padding:11px 14px;color:#294a40;font-size:1.01rem;font-weight:610;line-height:1.5;}
.bf-scenario-timeline{display:grid;grid-template-columns:1fr;gap:0;margin-top:7px;}
.bf-scenario-row{display:grid;grid-template-columns:42px 1fr;gap:8px;align-items:start;padding:10px 0;border-bottom:1px solid #eaf0e9;}
.bf-scenario-row:last-child{border-bottom:0;padding-bottom:2px;}
.bf-scenario-number{font-weight:760;font-size:.79rem;color:#1d8b75;letter-spacing:.01em;padding-top:2px;}
.bf-scenario-row strong{display:block;font-size:.9rem;color:#213f37;font-weight:760;margin-bottom:2px;}
.bf-scenario-row span{display:block;color:#50695e;font-size:.93rem;line-height:1.42;}
.bf-scenario-row:last-child span{font-weight:780;color:#09745e;}
.bf-scenario-footnote{font-size:.73rem;color:#73867e;border-top:1px solid #e6eee7;margin-top:13px;padding-top:11px;}
.bf-hero-caption{color:#597a6b;font-size:.82rem;margin-top:9px;}
.bf-story-rail{margin:15px 0 21px;display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.bf-story-part{background:#fff;border:1px solid #dfe9e1;border-radius:15px;padding:19px 21px;}
.bf-story-kicker{font-size:.76rem;font-weight:800;color:#0b7c68;letter-spacing:.08em;text-transform:uppercase;}
.bf-story-part h2{font-size:1.31rem!important;line-height:1.27;margin:9px 0!important;font-weight:790!important;letter-spacing:-.033em;}
.bf-story-part p{font-size:1rem!important;color:#537266!important;line-height:1.56!important;margin:0!important;}
.bf-story-meta{color:#607f70;font-size:.81rem;margin-top:11px;}
.bf-studio-heading{margin:19px 0 13px!important;}
.bf-studio-subtitle{font-size:1.06rem!important;max-width:825px!important;}
@media(max-width:980px){.bf-intro-grid{grid-template-columns:1fr;gap:20px;}.bf-scenario{max-width:720px;}.bf-hero-clear{padding:26px!important;}}
@media(max-width:690px){.bf-story-rail{grid-template-columns:1fr;}.bf-hero-clear h1{font-size:2.25rem!important;}.bf-scenario{padding:17px;}.bf-story-part{padding:16px;}}

/* Submission composition: keep the real Voice Studio close to the opening story. */
.bf-hero-clear{padding:26px 31px!important;}
.bf-hero-clear h1{font-size:clamp(2.32rem,3.15vw,3.3rem)!important;}
.bf-hero-sub{font-size:1.08rem!important;line-height:1.54!important;margin:14px 0 17px!important;}
.bf-scenario{padding:18px 20px;}
.bf-scenario h2{font-size:1.24rem!important;margin:7px 0 9px!important;}
.bf-scenario-row{padding:8px 0;}
.bf-scenario-footnote{margin-top:7px;padding-top:8px;}
.bf-story-part{padding:14px 17px;}
.bf-story-part h2{font-size:1.18rem!important;margin:6px 0!important;}
.bf-story-part p{font-size:.96rem!important;line-height:1.48!important;}
.bf-story-meta{font-size:.77rem;margin-top:8px;}
.bf-story-rail{margin:13px 0 16px;}

/* FINAL POLISH / native Streamlit buttons: text contrast including nested labels. */
[data-testid="stButton"] button[kind="primary"]:not(:disabled),
[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled){
 background:#087462!important;border:1px solid #087462!important;color:#ffffff!important;
 box-shadow:0 7px 18px rgba(4,95,77,.15)!important;font-weight:800!important;
 min-height:3.45rem!important;font-size:1.08rem!important;
}
[data-testid="stButton"] button[kind="primary"]:not(:disabled) *,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled) *{
 color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
 opacity:1!important;font-weight:800!important;
}
[data-testid="stButton"] button[kind="primary"]:not(:disabled):hover,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled):hover{
 background:#065849!important;border-color:#065849!important;transform:translateY(-1px);
}
[data-testid="stButton"] button[kind="primary"]:disabled,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:disabled{
 background:#e5efeb!important;border:1px solid #c7d9d1!important;
 color:#415d54!important;opacity:1!important;cursor:not-allowed!important;
}
[data-testid="stButton"] button[kind="primary"]:disabled *,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:disabled *{
 color:#415d54!important;-webkit-text-fill-color:#415d54!important;
 opacity:1!important;font-weight:750!important;
}
/* Lighter, recognizable capture workspace; keep native recording controls. */
[data-testid="stVerticalBlockBorderWrapper"]{
 background:linear-gradient(130deg,#f0f9f5 0%,#fcfffd 56%,#edf6f7 100%)!important;
 border:1px solid #bedfd4!important;border-radius:22px!important;
 box-shadow:0 12px 29px rgba(17,86,70,.06)!important;
}
[data-testid="stAudioInput"],[data-testid="stFileUploader"]{
 background:#fff!important;border:1px solid #bfddd2!important;border-radius:16px!important;
 box-shadow:0 5px 14px rgba(15,84,69,.05)!important;
}
[data-testid="stAudioInput"]:focus-within,[data-testid="stFileUploader"]:focus-within{
 outline:2px solid #84c9b7;outline-offset:2px;
}
.bf-studio-heading{background:linear-gradient(100deg,#e8f7ef,#eef8f7 72%,#f7fbf9);
 border:1px solid #d1e9dc;border-radius:19px;padding:22px 26px!important;margin:24px 0 18px!important;}
.bf-section-head{font-size:clamp(2.05rem,3.3vw,2.65rem)!important;line-height:1.15!important;}
.bf-studio-subtitle{font-size:1.14rem!important;line-height:1.54!important;max-width:850px!important;}
.bf-input-lead{font-size:1.55rem!important;line-height:1.3!important;}
.bf-input-help{font-size:1rem!important;line-height:1.55!important;}
.bf-triad{display:flex;gap:8px;flex-wrap:wrap;margin:7px 0 19px;}
.bf-triad-item{display:inline-flex;align-items:center;padding:8px 13px;border:1px solid #cce4d9;
 background:#fff;border-radius:999px;color:#136851;font-size:.9rem;font-weight:730;}
.bf-hero-clear .bf-overline{font-size:.81rem;}
.bf-hero-sub{font-size:1.17rem!important;max-width:890px!important;}
.bf-hero-sub strong{color:#124b3b;font-weight:790;}
.bf-hero-clear h1{font-size:clamp(2.65rem,3.6vw,3.95rem)!important;max-width:900px!important;}
.bf-hero-clear{padding:36px 42px!important;}
.bf-story-part h2{font-size:1.4rem!important;}
.bf-story-part p{font-size:1.07rem!important;}
.bf-story-meta{font-size:.85rem!important;line-height:1.5;}
@media(max-width:690px){
 .bf-hero-clear{padding:24px 21px!important;}.bf-hero-clear h1{font-size:2.35rem!important;}
 .bf-studio-heading{padding:19px!important;}
 .bf-triad-item{font-size:.83rem;padding:7px 10px;}
}
</style>""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def is_public_demo() -> bool:
    """Public deployments never load CPU-heavy Whisper or default server credits."""
    return (secret("BOLIFIX_PUBLIC_DEMO") or os.getenv("BOLIFIX_PUBLIC_DEMO", "false")).lower() == "true"


@st.cache_resource(show_spinner=False)
def locally_cached_whisper():
    """Load only already-downloaded model weights, without network access."""
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8", local_files_only=True)


def active_key() -> str:
    supplied = st.session_state.get("visitor_gnani_key", "").strip()
    if supplied:
        return supplied
    server_key = (secret("GNANI_API_KEY") or os.getenv("GNANI_API_KEY", "")).strip()
    if not server_key:
        return ""
    gate = (secret("BOLIFIX_DEMO_ACCESS_CODE") or os.getenv("BOLIFIX_DEMO_ACCESS_CODE", "")).strip()
    public = is_public_demo()
    if public and not gate:
        return ""  # Never expose unlimited server-side credits to the public by default.
    if gate and st.session_state.get("visitor_access_code", "") != gate:
        return ""
    return server_key


def call_prisma(data: bytes, filename: str, hint: str, fmt: str, key: str) -> str:
    """No persistent storage. Audio is sent solely to Gnani after user permission."""
    ext = Path(filename).suffix.lower()
    if ext not in EXTS:
        raise ValueError("Unsupported audio format.")
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("Please use an audio file under 10 MB.")
    if ext == ".wav":
        import wave
        try:
            with wave.open(io.BytesIO(data)) as wav:
                duration = wav.getnframes() / wav.getframerate()
        except (wave.Error, ZeroDivisionError, EOFError) as exc:
            raise ValueError("Invalid WAV audio.") from exc
        if duration > 30:
            raise ValueError("Please record under 30 seconds for this Prisma API.")
    response = requests.post(
        STT_URL,
        headers={"X-API-Key-ID": key},
        data={"model": "prisma-v3.0", "language_code": hint, "format": fmt},
        files={"audio_file": (filename, io.BytesIO(data))}, timeout=70,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Prisma could not complete this request (HTTP {response.status_code}).")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Prisma returned an invalid response.") from exc
    if not payload.get("success") or not isinstance(payload.get("transcript"), str):
        raise RuntimeError("Prisma did not return a usable transcript.")
    return payload["transcript"]


def call_timbre(question: str, language: str, key: str) -> bytes:
    from gnani.tts import GnaniTTSClient, AudioConfig
    sdk = GnaniTTSClient(api_key=key)
    audio = sdk.synthesize(
        question, model="timbre-v2.5", voice="Nalini" if language == "hi-IN" else "Kaveri",
        language=language, speed=1.0,
        audio_config=AudioConfig(sample_rate=16000, encoding="linear_pcm", container="wav"),
    )
    if not isinstance(audio, bytes) or len(audio) > 5 * 1024 * 1024 or not audio.startswith(b"RIFF"):
        raise RuntimeError("Timbre did not return expected WAV audio.")
    return audio


def clear_final_confirmation():
    st.session_state.pop("voice_confirmed", None)


def reset_speech_state():
    for key in ("voice_transcript", "voice_reply", "voice_decision", "voice_confirmed",
                "voice_timbre", "voice_second", "voice_second_decision", "voice_demo_flag",
                "voice_final_intent", "voice_final_checked", "voice_reply_sig", "voice_second_sig",
                "voice_whisper_text", "voice_whisper_sig"):
        st.session_state.pop(key, None)



def render_voice_progress(audio_present: bool) -> str:
    """Product-style tracker reflecting user actions, never invented milestones."""
    done = [audio_present,
            bool(st.session_state.get("voice_transcript")),
            bool(st.session_state.get("voice_reply")),
            bool(st.session_state.get("voice_confirmed"))]
    names = [("Capture", "Speaker audio"), ("Recognize", "Prisma hypothesis"),
             ("Recover", "Clarification reply"), ("Confirm", "Speaker-verified") ]
    parts = []
    for i, (label, detail) in enumerate(names):
        state = "bf-done" if done[i] else ("bf-current" if not any(not d for d in done[:i]) else "")
        glyph = "✓" if done[i] else f"{i + 1:02d}"
        parts.append(f'<div class="bf-journey-cell {state}"><span class="bf-num">{glyph}</span>'
                     f'<div><div class="bf-journey-label">{label}</div>'
                     f'<div class="bf-journey-description">{detail}</div></div></div>')
    return '<div class="bf-journey">' + ''.join(parts) + '</div>'


COMPARISON_SCENARIOS = {
    "Quantity mismatch": ("मुझे पाँच पैकेट चाहिए", "मुझे पंद्रह पैकेट चाहिए"),
    "Same meaning, different scripts": ("Panch nhi pandra packet chaiye", "पाँच नहीं पंद्रह पैकेट चाहिए"),
    "Negation difference": ("पंद्रह पैकेट नहीं चाहिए", "पंद्रह पैकेट चाहिए"),
    "Place name uncertainty": ("भंजनगर", "bhanjanagar"),
}


def load_comparison_scenario():
    """User-initiated, synthetic input only; never calls Prisma or Whisper."""
    key = st.session_state["cmp_scenario"]
    first, second = COMPARISON_SCENARIOS[key]
    st.session_state["cmp_prisma"] = first
    st.session_state["cmp_whisper"] = second
    st.session_state.pop("cmp_result", None)



key_state = "Gnani ready" if active_key() else "API key required for live demo"
st.markdown('''<div class="bf-brand"><div class="bf-brand-name"><span class="bf-brand-glyph">B</span>BOLIFIX</div>
<div class="bf-brand-right">Speech recovery for Indian languages &nbsp;·&nbsp; Built with Gnani</div></div>''',unsafe_allow_html=True)

st.markdown('''<section class="bf-hero-clear" aria-label="What BOLIFIX does">
<div class="bf-overline">BOLIFIX / Voice recovery + verified learning</div>
<h1>Recover the meaning now.<br><span>Learn for the next conversation.</span></h1>
<div class="bf-hero-sub"><strong>BOLIFIX helps Indian-language voice applications recover misunderstood instructions before acting.</strong> It asks a focused question and lets the speaker confirm. With separate permission, human-reviewed recordings can help an independent speech model learn and adapt for future calls.</div>
<div class="bf-triad" aria-label="Three parts of the BOLIFIX vision">
<span class="bf-triad-item">Recover important details</span>
<span class="bf-triad-item">Learn from verified corrections</span>
<span class="bf-triad-item">Adapt after evaluation</span>
</div>
<div class="bf-hero-actions"><a class="bf-hero-cta" href="#voice-studio">Try voice recovery &nbsp; ↗</a></div>
<div class="bf-hero-caption">Designed for Indian-language voice services. Live demo tested primarily with Hindi and Hinglish; other language options require validation.</div>
</section>
<div class="bf-story-rail" aria-label="BOLIFIX recovery and learning">
<div class="bf-story-part"><div class="bf-story-kicker">In this conversation</div>
<h2>Correct the instruction before it matters.</h2><p>Prisma recognizes speech. BOLIFIX flags a possible problem. Timbre asks a short question. The speaker confirms what they meant.</p>
<div class="bf-story-meta">Working short-recording experience · It cannot catch every error.</div></div>
<div class="bf-story-part"><div class="bf-story-kicker">Across future conversations</div>
<h2>Build better recognition from real mistakes.</h2><p>With separate consent, reviewers check the original audio. Trusted examples can then be used to train and evaluate an independent speech model.</p>
<div class="bf-story-meta">Local review tools work today; model retraining and cross-app rollout are not yet live.</div></div>
</div>''', unsafe_allow_html=True)

with st.expander("See an example: why a single number matters", expanded=False):
    st.markdown('''<div class="bf-scenario" aria-label="Illustrative voice-order example">
<div class="bf-scenario-kicker">Illustrative example · not a live AI result</div>
<h2>One missed number can change an order.</h2>
<div class="bf-scenario-quote">Caller: “Not five. Fifteen packets.”</div>
<div class="bf-scenario-timeline">
<div class="bf-scenario-row"><div class="bf-scenario-number">01</div><div><strong>Possible recognition mistake</strong><span>The voice app hears “5 packets.”</span></div></div>
<div class="bf-scenario-row"><div class="bf-scenario-number">02</div><div><strong>BOLIFIX asks for clarity</strong><span>“Is the final quantity 5 or 15?”</span></div></div>
<div class="bf-scenario-row"><div class="bf-scenario-number">03</div><div><strong>The speaker decides</strong><span>“15 packets” — confirmed, not guessed.</span></div></div>
</div><div class="bf-scenario-footnote">This is a hypothetical scenario, not a recorded Prisma result. Try the live Voice Studio below.</div>
</div>''', unsafe_allow_html=True)

session_slot = st.empty()  # Filled after Voice Studio actions, so state is never one rerun behind.

voice_tab, compare_tab, learning_tab, system_tab = st.tabs(["Voice Studio", "Model comparison", "Learning loop", "Technology"])

with voice_tab:
    st.markdown('''<div class="bf-studio-heading" id="voice-studio"><div>
    <div class="bf-section-head">Voice Recovery Studio</div>
    <div class="bf-studio-subtitle">Record or upload an instruction. Review what Prisma heard, clarify an important detail if needed, and confirm the final meaning.</div>
    </div><span class="bf-studio-badge">Live voice recovery</span></div>''', unsafe_allow_html=True)
    progress_slot = st.empty()
    st.markdown('<div class="bf-privacy-note">A recording is sent to Gnani only after you give permission and press Transcribe. This demo does not save your voice.</div>',unsafe_allow_html=True)
    with st.expander("API access" if active_key() else "Connect Gnani to run a live demo", expanded=not bool(active_key())):
        st.text_input("Your Gnani API key (session only)", type="password", key="visitor_gnani_key",
                      help="For a public deployment, visitors bring their own key. Never post an API key in GitHub or a video.")
        if (secret("BOLIFIX_DEMO_ACCESS_CODE") or os.getenv("BOLIFIX_DEMO_ACCESS_CODE")):
            st.text_input("Private demo access code (optional)", type="password", key="visitor_access_code")
        if active_key():
            st.success("Live Gnani access is available for this session.")
        else:
            st.info("No live API access yet. You can still explore the comparison tab with sample text.")
    with st.container(border=True):
        st.markdown('<div class="bf-studio-kicker">STEP 01 / YOUR VOICE</div><div class="bf-input-lead">Start with a voice instruction</div>'
                    '<div class="bf-input-help">Record or upload a short clip. Choose your language preferences below, grant permission, then transcribe.</div>',
                    unsafe_allow_html=True)
        input_mode = st.radio(
            "Choose how to provide audio", ["Record with microphone", "Upload an audio file"],
            horizontal=True, key="voice_input_mode",
        )
        microphone = None
        upload = None
        if input_mode == "Record with microphone":
            microphone = st.audio_input("Record your instruction", key="voice_mic")
        else:
            upload = st.file_uploader(
                "Choose a recording · WAV, MP3, OGG, FLAC, AAC or M4A · max 10 MB",
                type=["wav","mp3","ogg","flac","aac","m4a"], key="voice_upload",
            )
        src = microphone or upload
        with st.expander("Language & transcription preferences", expanded=False):
            hint_col,style_col = st.columns(2)
            with hint_col:
                hint = st.selectbox("Language hint", ["indic-auto", "hi-IN,en-IN", "hi-IN", "en-IN", "bn-IN", "mr-IN", "gu-IN", "pa-IN", "kn-IN", "te-IN", "ta-IN", "ml-IN"], index=0)
            with style_col:
                fmt = st.selectbox("Text output", ["codemix","verbatim","transcribe","transliterate"])
        st.markdown('<div class="bf-privacy-note">Short recordings only · No automatic recording or model training · The button becomes available after you provide audio, give permission, and connect Gnani.</div>', unsafe_allow_html=True)
        consent = st.checkbox("I am the speaker or have permission to send this audio to Gnani for transcription.", key="voice_consent")
        current_sig = hashlib.sha256(src.getvalue()).hexdigest() if src else None
        if st.session_state.get("voice_audio_sig") != current_sig:
            reset_speech_state()
            st.session_state["voice_audio_sig"] = current_sig
        if st.button("Transcribe with Prisma", type="primary", disabled=not (src and consent and active_key()), use_container_width=True):
            try:
                with st.spinner("Prisma is analyzing your audio..."):
                    transcript = call_prisma(src.getvalue(), getattr(src,"name","microphone.wav"),hint,fmt,active_key())
                reset_speech_state()
                st.session_state["voice_transcript"] = transcript
                st.session_state["voice_hint"] = hint
                st.session_state["voice_format"] = fmt
            except (ValueError, RuntimeError, requests.RequestException, ImportError) as exc:
                st.error(f"Transcription failed: {type(exc).__name__}: {str(exc)[:180]}")
    if st.session_state.get("voice_transcript"):
        original = st.session_state["voice_transcript"]
        st.markdown('<div class="bf-step">STEP 02 / RECOGNITION AND RISK REVIEW</div>',unsafe_allow_html=True)
        st.markdown('**Recognized by Prisma v3.0**')
        st.markdown(f'<div class="bf-transcript">{__import__("html").escape(original)}</div>',unsafe_allow_html=True)
        st.markdown("**Optional: compare with local Whisper**")
        if not is_public_demo():
            st.caption("Local Whisper Small is an optional research check on the same recording. "
                       "It uses a previously downloaded model; no Gnani call or automatic download.")
            if st.button("Check the same audio with local Whisper", key="voice_local_whisper"):
                try:
                    with st.spinner("Running local Whisper Small on CPU..."):
                        other = run_local_whisper(src.getvalue(), locally_cached_whisper())
                    st.session_state["voice_whisper_text"] = other
                    st.session_state["voice_whisper_sig"] = current_sig
                except (ImportError, RuntimeError, OSError, ValueError, TypeError) as exc:
                    st.error("Local Whisper is unavailable. Install faster-whisper and cache "
                             "Whisper Small locally; the voice-recovery path still works without it. "
                             f"({type(exc).__name__})")
        else:
            st.caption("Independent Whisper analysis runs only in the local research installation, "
                       "not on the public demo server.")
        whisper_text = st.session_state.get("voice_whisper_text", "") if st.session_state.get("voice_whisper_sig") == current_sig else ""
        dual_report = None
        if whisper_text:
            st.markdown("**Whisper Small · independent hypothesis**")
            st.write(whisper_text)
            dual_report = compare_hypotheses(original, whisper_text)
            if dual_report["potential_high_impact_difference"]:
                st.warning("Models disagree on potentially important content. "
                           "Confirm with the speaker; neither model is assumed correct.")
            elif dual_report["limited_equivalent"]:
                st.success("Limited known wording equivalence detected; this does not prove accuracy.")
            else:
                st.info("Comparison needs listening review; script differences can cause false alarms.")
        analysis = suggest_clarification(original)
        if analysis["needs_clarification"] or (dual_report and dual_report["clarification_plan"]["can_speak"]):
            if analysis["signals"]:
                st.caption("Potentially consequential language detected: " + ", ".join(analysis["signals"]))
            spoken_lang = st.selectbox("Clarification voice",["hi-IN","en-IN"],format_func=lambda v:"Hindi" if v=="hi-IN" else "English",key="voice_tts_lang")
            route = dual_report["clarification_plan"] if dual_report else None
            question = ((route["question_hi"] if spoken_lang == "hi-IN" else route["question_en"])
                        if route and route["can_speak"] else clarification_question(original,spoken_lang))
            if question:
                st.markdown("**Follow-up question**")
                st.info(question)
                if st.button("Generate spoken question with Timbre", disabled=not bool(active_key()), key="voice_speak"):
                    try:
                        with st.spinner("Generating a voice response..."):
                            st.session_state["voice_timbre"] = call_timbre(question,spoken_lang,active_key())
                    except Exception as exc:
                        st.error(f"Timbre could not generate audio ({type(exc).__name__}). Check access and TTS permissions.")
                if st.session_state.get("voice_timbre"):
                    st.audio(st.session_state["voice_timbre"], format="audio/wav")
            st.markdown('<div class="bf-step">STEP 03 / FOCUSED CLARIFICATION</div>',unsafe_allow_html=True)
            reply_file = st.audio_input("Answer the clarification by voice",key="voice_reply_mic")
            new_reply_sig = hashlib.sha256(reply_file.getvalue()).hexdigest() if reply_file else None
            if new_reply_sig != st.session_state.get("voice_reply_sig"):
                st.session_state["voice_reply_sig"] = new_reply_sig
                for old in ("voice_reply", "voice_decision", "voice_second", "voice_second_decision", "voice_confirmed", "voice_final_intent", "voice_final_checked"):
                    st.session_state.pop(old, None)
            reply_consent = st.checkbox("I have permission to send this reply audio to Gnani.",key="voice_reply_consent")
            if st.button("Transcribe reply",disabled=not(reply_file and reply_consent and active_key()),key="voice_transcribe_reply"):
                try:
                    with st.spinner("Listening to the clarification..."):
                        reply = call_prisma(reply_file.getvalue(),getattr(reply_file,"name","reply.wav"),st.session_state.get("voice_hint","indic-auto"),st.session_state.get("voice_format","codemix"),active_key())
                    st.session_state["voice_reply"] = reply
                    st.session_state["voice_decision"] = next_recovery_step(original,reply,1)
                    st.session_state.pop("voice_confirmed",None)
                    st.session_state.pop("voice_second_decision",None)
                    st.session_state.pop("voice_final_intent",None)
                    st.session_state.pop("voice_final_checked",None)
                except (ValueError,RuntimeError,requests.RequestException,ImportError) as exc:
                    st.error(f"Reply failed: {type(exc).__name__}: {str(exc)[:160]}")
            if st.session_state.get("voice_reply"):
                st.markdown("**Recognized reply**")
                st.write(st.session_state["voice_reply"])
                decision = st.session_state.get("voice_second_decision") or st.session_state.get("voice_decision")
                st.caption(decision["message"])
                if decision["state"] == "ask_followup":
                    st.warning("One final focused attempt is available. Otherwise use manual verification.")
                    second = st.audio_input("Record one final number only",key="voice_second_mic")
                    second_sig = hashlib.sha256(second.getvalue()).hexdigest() if second else None
                    if second_sig != st.session_state.get("voice_second_sig"):
                        st.session_state["voice_second_sig"] = second_sig
                        for old in ("voice_second", "voice_second_decision", "voice_confirmed", "voice_final_intent", "voice_final_checked"):
                            st.session_state.pop(old, None)
                    second_consent=st.checkbox("I have permission to send the second reply to Gnani.",key="voice_second_consent")
                    if st.button("Transcribe final reply",disabled=not(second and second_consent and active_key()),key="voice_second_btn"):
                        try:
                            final_reply=call_prisma(second.getvalue(),getattr(second,"name","second.wav"),st.session_state.get("voice_hint","indic-auto"),st.session_state.get("voice_format","codemix"),active_key())
                            st.session_state["voice_second"] = final_reply
                            st.session_state["voice_second_decision"] = next_recovery_step(original,final_reply,2)
                            st.session_state.pop("voice_confirmed",None)
                            st.session_state.pop("voice_final_intent",None)
                            st.session_state.pop("voice_final_checked",None)
                        except (ValueError,RuntimeError,requests.RequestException,ImportError) as exc:
                            st.error(f"Final reply failed: {type(exc).__name__}: {str(exc)[:160]}")
                if st.session_state.get("voice_second"):
                    st.caption("Second reply recognized: " + st.session_state["voice_second"])
                decision = st.session_state.get("voice_second_decision") or st.session_state.get("voice_decision")
                proposed = decision.get("candidate")
                if proposed is not None:
                    st.info(f"Possible intended value: {proposed} — not verified yet.")
                confirmed_text = st.text_input("Final intended value (speaker may edit)",value=str(proposed or ""),key="voice_final_intent",on_change=clear_final_confirmation)
                checked = st.checkbox("I personally verify this is the speaker's intended value.",key="voice_final_checked")
                if st.button("Confirm intended value",disabled=not(checked and confirmed_text.strip()),key="voice_final_btn"):
                    st.session_state["voice_confirmed"] = confirmed_text.strip()
                if st.session_state.get("voice_confirmed"):
                    st.markdown('<div class="bf-step">RESULT / SPEAKER-VERIFIED OUTCOME</div>',unsafe_allow_html=True)
                    st.markdown(f'<div class="bf-result"><div class="bf-result-eyebrow">✓ Speaker-confirmed outcome</div>'
                                f'<div class="bf-result-value">{__import__("html").escape(st.session_state["voice_confirmed"])}</div>'
                                f'<div class="bf-result-note">Confirmed by a person, not assumed by either AI model.</div>'
                                f'</div>', unsafe_allow_html=True)
                    st.caption("Session only. This confirms intended meaning, not the exact original-audio words. "
                               "No order was placed and no ASR model was trained.")
                    st.markdown('''<div class="bf-edu-card" style="margin-top:13px;background:#eef7f2;box-shadow:none">
                    <div class="bf-edu-kicker">WHAT HAPPENS AFTER CONFIRMATION?</div>
                    <h3>Recovered for this conversation. Not automatically learned by the model.</h3>
                    <p>Only the current browser session has the confirmed value. A private JSON receipt is optional.
                    Future training needs separate recording consent, two independent reviewers, enough diverse speakers, and evaluation.</p>
                    </div>''',unsafe_allow_html=True)
                    if src:
                        receipt = make_receipt(
                            original=original,
                            first_reply=st.session_state.get("voice_reply", ""),
                            second_reply=st.session_state.get("voice_second", ""),
                            intended_value=st.session_state["voice_confirmed"],
                            original_audio=src.getvalue(),
                            whisper_hypothesis=whisper_text,
                        )
                        st.download_button(
                            "Download private recovery receipt (JSON)",
                            data=receipt_json(receipt), file_name="bolifix-recovery.json",
                            mime="application/json", key="voice_receipt_download",
                            help="Contains your transcripts and confirmed value; do not share if sensitive.",
                        )
        else:
            st.success("No predefined high-impact term was flagged. This does not prove that recognition was perfect.")

    progress_slot.markdown(render_voice_progress(bool(src)), unsafe_allow_html=True)

# Update the page-level session indicator from the same run as the voice actions.
session_progress = session_learning_status(
    audio_present=bool(st.session_state.get("voice_audio_sig")),
    transcript_present=bool(st.session_state.get("voice_transcript")),
    reply_present=bool(st.session_state.get("voice_reply") or st.session_state.get("voice_second")),
    confirmed_intent=bool(st.session_state.get("voice_confirmed")),
)
if session_progress["speaker_confirmed"]:
    session_text = "Speaker-confirmed intent ready in this session · no model weights updated"
elif session_progress["runtime"][2]["complete"]:
    session_text = "Reply received · awaiting speaker verification"
elif session_progress["runtime"][1]["complete"]:
    session_text = "Prisma hypothesis ready · review or clarify"
elif session_progress["runtime"][0]["complete"]:
    session_text = "Audio provided · ready for transcription"
else:
    session_text = "Ready for a permitted voice recording"
key_state = "Gnani key configured" if active_key() else "Explore locally without API calls"
session_slot.markdown(f'''<div class="bf-session-indicator"><div><span class="bf-session-label">YOUR LIVE SESSION</span> &nbsp;
<span class="bf-session-detail">{session_text}</span></div><span class="bf-pill"><span class="bf-pill-dot"></span>{key_state}</span></div>''', unsafe_allow_html=True)

with compare_tab:
    st.markdown('<div class="bf-section-head">Compare two recognizers</div>', unsafe_allow_html=True)
    st.markdown('<p class="bf-muted">Explore disagreements between two speech-recognition hypotheses. The comparison is rule-based; it never chooses a winner without a speaker.</p>', unsafe_allow_html=True)
    st.markdown('<div class="bf-step">Interactive example lab · no API calls</div>', unsafe_allow_html=True)
    preset_col, load_col = st.columns([2.5, 1], gap="medium", vertical_alignment="bottom")
    with preset_col:
        st.selectbox("Choose a scenario", options=list(COMPARISON_SCENARIOS), key="cmp_scenario")
    with load_col:
        st.button("Load scenario", on_click=load_comparison_scenario,
                  key="cmp_load_scenario", use_container_width=True)
    st.caption("The examples are synthetic and explicitly labeled. Load one to fill the fields, then analyze it.")
    st.session_state.setdefault("cmp_prisma", "मुझे पाँच पैकेट चाहिए")
    st.session_state.setdefault("cmp_whisper", "मुझे पंद्रह पैकेट चाहिए")
    l,r=st.columns(2,gap="large")
    with l:
        if st.session_state.get("voice_transcript") and st.button("Use latest Prisma transcript",key="copy_prisma"):
            st.session_state["cmp_prisma"] = st.session_state["voice_transcript"]
        a=st.text_area("Prisma hypothesis",height=140,key="cmp_prisma")
    with r:
        if st.session_state.get("voice_whisper_text") and st.button("Use latest local Whisper result",key="copy_whisper"):
            st.session_state["cmp_whisper"] = st.session_state["voice_whisper_text"]
        b=st.text_area("Whisper hypothesis (manually supplied or local)",height=140,key="cmp_whisper")
    if st.button("Analyze disagreement",type="primary",disabled=not(a.strip() and b.strip()),key="cmp_btn"):
        st.session_state["cmp_result"] = (a,b,compare_hypotheses(a,b))
    data = st.session_state.get("cmp_result")
    if data and data[:2]==(a,b):
        report=data[2]
        high=report["potential_high_impact_difference"]
        if high: st.error("Priority review: potentially consequential disagreement")
        elif report["limited_equivalent"]: st.success("Limited equivalent wording detected — still not proof of correctness")
        else: st.warning("Manual listening recommended")
        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**Prisma number mentions**")
            st.write(", ".join(report["prisma_numbers"]) or "None detected")
        with col_b:
            st.markdown("**Whisper number mentions**")
            st.write(", ".join(report["whisper_numbers"]) or "None detected")
        plan=report["clarification_plan"]
        st.markdown("**Suggested next action**")
        st.info(plan.get("question_en") or "Listen to the original audio and review manually. No automatic question recommended.")
        st.caption("These results are rule-based signals, not a verdict on which model is correct or a WER accuracy score.")
        if plan["route"] == "confirm_final_number":
            reply_text=st.text_input("Try an example clarification reply (text-only simulation)",value="pandrah",key="cmp_reply")
            if reply_text.strip():
                decision=next_dual_recovery_step(a,b,reply_text,1)
                st.caption("Simulated state: " + decision["state"])
                if decision["candidate"] is not None:
                    st.write("Proposed number for speaker confirmation:",decision["candidate"])
        st.caption("To run both models on an approved recording, use the local scripts/dual_asr.py command documented in the repository.")
    else:
        st.caption("Select Analyze disagreement to see the comparison results.")

with learning_tab:
    st.markdown('<div class="bf-step">BOLIFIX / Continual intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="bf-section-head">The correction is the beginning, not the end.</div>', unsafe_allow_html=True)
    st.markdown('<p class="bf-muted">A call has two outcomes: an instruction is recovered <b>now</b>; a future model may improve <b>later</b>—only when the original audio is separately permitted, reviewed, and evaluated.</p>', unsafe_allow_html=True)
    path = st.radio("Follow the intelligence lifecycle", ["01 · During the conversation", "02 · After approved review", "03 · Across future applications"],
                    horizontal=True, key="bf_learning_path")
    learning_story = {
        "01 · During the conversation": ("WORKING WITH SHORT CLIPS", "Resolve the current instruction", "Prisma recognizes the voice. BOLIFIX flags specific potential mistakes. Timbre can ask a follow-up. The speaker explicitly confirms their intended meaning, so the CURRENT interaction can be corrected without pretending either recognizer is always right."),
        "02 · After approved review": ("LOCAL RESEARCH TOOLS", "Turn consented mistakes into trustworthy evidence", "Retaining audio requires a separate opt-in. In the private research application, two independent reviewers hear the ORIGINAL recording, establish its actual word-for-word transcript, and validate permission and speaker separation. A confirmed quantity alone is not a speech-training label."),
        "03 · Across future applications": ("FUTURE DEVELOPMENT", "Learn once, improve where permission allows", "After enough authorized evidence exists, an independently trainable ASR model could be fine-tuned, tested on untouched speakers, then cautiously deployed to other participating voice applications. Shared databases, live-phone adapters, and automatic training are not running today. Prisma's own model can only be updated by its provider."),
    }
    story_label, story_title, story_content = learning_story[path]
    st.markdown(f'<div class="bf-edu-card"><div class="bf-edu-kicker">{story_label}</div>'
                f'<h3>{story_title}</h3><p>{story_content}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="bf-section-head">Recover now. Learn from verified evidence.</div>', unsafe_allow_html=True)
    st.markdown('<p class="bf-muted">BOLIFIX is designed as a recovery-and-learning layer for Indian-language voice applications. '
                'It distinguishes an immediate conversation correction from the slower, permission-controlled process of improving a trainable ASR model.</p>',
                unsafe_allow_html=True)
    live = session_learning_status(
        audio_present=bool(st.session_state.get("voice_audio_sig")),
        transcript_present=bool(st.session_state.get("voice_transcript")),
        reply_present=bool(st.session_state.get("voice_reply") or st.session_state.get("voice_second")),
        confirmed_intent=bool(st.session_state.get("voice_confirmed")),
    )
    st.markdown('<div class="bf-learning-banner"><div class="bf-learning-eyebrow">CURRENT BROWSER SESSION · REAL EVENTS ONLY</div>'
                '<div class="bf-learning-headline">Correction is immediate. Model training is deliberate.</div>'
                '<p>A confirmed value can resolve the current interaction. A voice model changes only after separate '
                'recording permissions, independent word-for-word review, dataset checks, training and evaluation.</p></div>',
                unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown('**Live session journey**')
        for stage in live["runtime"]:
            klass = "bf-learn-complete" if stage["complete"] else "bf-learn-idle"
            glyph = "✓" if stage["complete"] else "·"
            st.markdown(
                f'<div class="bf-learning-stage {klass}"><div class="bf-learning-icon">{glyph}</div>'
                f'<div><strong>{stage["label"]}</strong><div class="bf-learning-desc">{stage["detail"]}</div></div></div>',
                unsafe_allow_html=True,
            )
        if live["speaker_confirmed"]:
            st.success("Speaker-confirmed intended value exists in this browser session. This is immediate task recovery, not model training.")
        else:
            st.info("Complete the Voice studio flow to see real session milestones change here. Nothing is fabricated for the demo.")
    with right:
        st.markdown('**Beyond this session**')
        for title, status, explanation in LEARNING_STAGES:
            st.markdown(f'<div class="bf-learning-future"><strong>{title}</strong>'
                        f'<span class="bf-learning-tag">{status}</span>'
                        f'<div class="bf-learning-desc">{explanation}</div></div>',
                        unsafe_allow_html=True)
    st.markdown('<div class="bf-line"></div>', unsafe_allow_html=True)
    st.markdown('**Explore the learning architecture**')
    learning_focus = st.selectbox(
        "Select a learning mechanism",
        ["During a call · immediate recovery", "After a call · approved evidence", "Human verification · true speech labels",
         "Model improvement · fine-tuning", "Across applications · collaborative learning", "Live phone integration · streaming"],
        key="learning_focus",
    )
    learning_details = {
        "During a call · immediate recovery":
            "Working prototype: Prisma provides a first transcript, BOLIFIX asks when it spots certain risk signals, and the speaker confirms the intended action. It updates that interaction's final decision; it does not change Prisma's model weights.",
        "After a call · approved evidence":
            "Research tools implemented locally: with separate explicit permission, the private backend can retain a recording and recovery evidence. The public Showcase does not save recordings or send events to that backend automatically.",
        "Human verification · true speech labels":
            "Local research tools implemented: the original audio can be listened to by two separate reviewers to establish word-for-word reference text. Confirming an order quantity is NOT enough to create a transcription training label.",
        "Model improvement · fine-tuning":
            "Not yet trained: after enough approved recordings from genuinely different speakers are available, an independently trainable open ASR model can be fine-tuned and compared with its unchanged baseline on untouched test audio. No accuracy improvement has been demonstrated.",
        "Across applications · collaborative learning":
            "Planned research: separate authorized applications would contribute independently reviewed examples to an approved training pool, respecting organization boundaries and permissions. No live cross-app pooling or federated training exists today.",
        "Live phone integration · streaming":
            "Planned integration: connect a real authorized phone/voice-agent audio stream through a documented streaming ASR interface. The current working Voice studio handles short recorded/uploaded clips, not live telephone calls.",
    }
    st.markdown(f'<div class="bf-learning-explainer">{learning_details[learning_focus]}</div>', unsafe_allow_html=True)
    st.caption("Learned intent is distinct from learned audio transcription. Public Showcase: session-only by design. "
               "Local research backend: private evidence and review, with separate permission. No automatic retraining or cross-app sharing.")

with system_tab:
    st.markdown('<div class="bf-step">BOLIFIX / System architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="bf-section-head">A recovery layer between speech AI and real-world actions.</div>', unsafe_allow_html=True)
    st.markdown('<p class="bf-muted">Speech-to-text alone can produce plausible but consequential mistakes. BOLIFIX adds a speaker-controlled correction loop and a governed path from approved evidence to future models.</p>', unsafe_allow_html=True)
    left_sys, right_sys = st.columns([1.12,.88], gap="large")
    with left_sys:
        for eyebrow,title,body in [
            ("01 / REAL-TIME VALUE", "Recover the meaning before acting", "A correction, negation or quantity can change an order or reservation. BOLIFIX flags supported signals and asks the speaker directly; it never creates real orders or performs transactions in this demo."),
            ("02 / TWO MODELS, ONE DECISION", "Use AI suggestions, keep the speaker in charge", "Gnani Prisma handles primary recognition; local Whisper Small can provide an independent second hypothesis. Agreement is not proof, and the speaker must confirm the critical meaning."),
            ("03 / PRIVACY BY DESIGN", "Data collection is not automatic learning", "The Showcase processes short clips in memory. A separate local research application can retain permitted evidence, conduct two-reviewer word-for-word transcription, and validate potential datasets. The two applications are not connected for automatic retention."),
            ("04 / LONG-TERM PLATFORM", "An upgrade path across applications", "The proposed shared-learning network would combine independently approved and reviewed evidence, fine-tune an open ASR model, evaluate it on unseen speakers, and roll out new versions with controls. Live phone streaming and model rollout remain engineering work."),
        ]:
            st.markdown(f'<div class="bf-edu-card"><div class="bf-edu-kicker">{eyebrow}</div><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    with right_sys:
        st.markdown("""<div class="bf-system-panel"><div class="bf-edu-kicker" style="color:#a6edca">TECHNOLOGY MAP</div>
<h3>Intelligence with accountability.</h3><p>Each component has one job. The strongest claim is what we can demonstrate with a real voice.</p>
<div class="bf-system-row"><div class="bf-system-dot">P</div><div><strong>Prisma v3.0 · primary ASR</strong><span>Converts a consented recording into a transcript hypothesis.</span></div></div>
<div class="bf-system-row"><div class="bf-system-dot">B</div><div><strong>BOLIFIX · risk and recovery</strong><span>Detects supported ambiguity patterns, plans clarification and tracks explicit confirmation.</span></div></div>
<div class="bf-system-row"><div class="bf-system-dot">T</div><div><strong>Timbre v2.5 · voice clarification</strong><span>Speaks a question only after user action, with no automatic credit usage.</span></div></div>
<div class="bf-system-row"><div class="bf-system-dot">W</div><div><strong>Whisper Small · optional independent ASR</strong><span>Runs locally on cached weights for exploratory comparison, not as a certified benchmark.</span></div></div>
<div class="bf-system-row"><div class="bf-system-dot">↻</div><div><strong>Private learning infrastructure · research</strong><span>Approved evidence, two-reviewer labels, dataset validation, and evaluation tools.</span></div></div>
</div>""", unsafe_allow_html=True)
        with st.expander("What has not yet shipped?"):
            st.markdown("**Not yet live:** automatic phone-call ingestion, cross-application/shared-database learning, open-ASR fine-tuning with verified improvement, model promotion/rollback, and production-grade multi-tenant security. Gnani Prisma's internal model is not modified by BOLIFIX.")
            st.markdown("**Accuracy:** one approved recording from one speaker does not establish a benchmark or reliable improvement claim.")
    st.markdown('<div class="bf-narrative-header"><div class="bf-narrative-title">The current stack</div><div class="bf-narrative-detail">Real interfaces · documented research boundaries</div></div>', unsafe_allow_html=True)
    cols=st.columns(4)
    for col,(name,desc) in zip(cols,[
        ("Prisma v3.0","Gnani speech-to-text"),("Timbre v2.5","Gnani text-to-speech"),
        ("Streamlit","Demo and research UI"),("FastAPI + SQLite","Private local evidence tools")]):
        with col:
            st.markdown(f'<div class="bf-metric"><b>{name}</b><small>{desc}</small></div>',unsafe_allow_html=True)

st.markdown('<div class="bf-line"></div>',unsafe_allow_html=True)
st.markdown('<div class="bf-foot">BOLIFIX · Live short-clip recovery. Private evidence and review tools run separately. Live telephone integration, automatic model training, and cross-application rollout are not yet available.</div>',unsafe_allow_html=True)
