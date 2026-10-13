import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="SupportPilot", page_icon="🎧", layout="wide")
st.title("🎧 SupportPilot — agentic ticket triage with human-in-the-loop")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", timeout=300, **kw)
    if not r.ok:
        st.error(r.text)
        return None
    return r.json()


t_kb, t_new, t_queue, t_stats, t_ml = st.tabs(
    ["1 Knowledge base", "2 New ticket", "3 Review queue", "4 Analytics", "5 ML model"])

with t_kb:
    files = st.file_uploader("Upload help-center articles (TXT / MD / PDF / DOCX)", accept_multiple_files=True)
    if st.button("Ingest", disabled=not files):
        with st.spinner("Indexing into Pinecone…"):
            res = api("POST", "/ingest", files=[("files", (f.name, f.getvalue())) for f in files])
        if res:
            st.success(res)

with t_new:
    text = st.text_area("Customer ticket", height=130,
                        placeholder="Where is my order ORD-1001? It has not arrived yet.")
    if st.button("Triage ticket", disabled=not text.strip()):
        with st.spinner("Agent working…"):
            d = api("POST", "/tickets", json={"text": text})
        if d:
            icon = "✅ auto-sent" if d["decision"] == "auto_send" else "🧑‍💼 sent to human review"
            (st.success if d["decision"] == "auto_send" else st.warning)(f"Ticket #{d['ticket_id']}: {icon}")
            c = st.columns(4)
            c[0].metric("Category", d["category"] or "-", f"conf {d['category_conf']}")
            c[1].metric("Priority", d["priority"] or "-")
            c[2].metric("Sentiment", d["sentiment"] or "-", d["sentiment_score"])
            c[3].metric("Groundedness", d["groundedness"] if d["groundedness"] is not None else "-")
            for r in d["reasons"]:
                st.write("• " + r)
            if d["flags"]:
                st.info("Governance flags: " + ", ".join(d["flags"]))
            st.subheader("Draft reply")
            st.write(d["draft"] or "_No draft generated_")
            with st.expander("Sources"):
                for i, s in enumerate(d["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']}** ({s['score']})")
                    st.caption(s["text"])
            with st.expander("Agent trace"):
                st.dataframe(d["trace"], use_container_width=True)

with t_queue:
    pending = api("GET", "/tickets?status=pending") or []
    st.caption(f"{len(pending)} ticket(s) awaiting review")
    for t in pending:
        with st.expander(f"#{t['id']} · {t['category']} · {t['priority']} priority · {t['sentiment']}"):
            st.markdown(f"**Ticket:** {t['text']}")
            st.warning(t["reasons"])
            reply = st.text_area("Reply to send", t["draft"] or "", key=f"reply{t['id']}", height=140)
            note = st.text_input("Reviewer note", key=f"note{t['id']}")
            a, b = st.columns(2)
            if a.button("✅ Approve & send", key=f"ok{t['id']}"):
                api("POST", f"/tickets/{t['id']}/review", json={"approve": True, "final_reply": reply, "note": note})
                st.rerun()
            if b.button("❌ Reject", key=f"no{t['id']}"):
                api("POST", f"/tickets/{t['id']}/review", json={"approve": False, "final_reply": None, "note": note})
                st.rerun()

with t_stats:
    m = api("GET", "/metrics")
    if m and m["tickets"]:
        c = st.columns(5)
        c[0].metric("Tickets", m["tickets"])
        c[1].metric("Auto-resolved", f"{m['auto_rate']:.0%}")
        c[2].metric("Pending review", m["pending"])
        c[3].metric("Avg groundedness", m["avg_groundedness"])
        c[4].metric("Avg latency", f"{m['avg_latency_ms']:.0f} ms")
        a, b, c3 = st.columns(3)
        a.subheader("By category"); a.bar_chart(pd.Series(m["by_category"]))
        b.subheader("By priority"); b.bar_chart(pd.Series(m["by_priority"]))
        c3.subheader("By sentiment"); c3.bar_chart(pd.Series(m["by_sentiment"]))
        st.subheader("Recent tickets")
        rows = pd.DataFrame(api("GET", "/tickets?limit=50") or [])
        st.dataframe(rows[["id", "category", "priority", "sentiment", "groundedness", "decision", "status", "reasons"]],
                     use_container_width=True)
    else:
        st.info("No tickets yet.")

with t_ml:
    st.caption("TF-IDF + logistic regression trained on data/tickets.csv (synthetic demo data).")
    if st.button("Train / retrain"):
        st.json(api("POST", "/ml/train"))
    st.json(api("GET", "/ml/info"))
    probe = st.text_input("Try the classifier")
    if probe:
        st.json(api("POST", "/ml/predict", json={"text": probe}))
