import streamlit as st
from typing import Generator
from groq import Groq
import fitz  # PyMuPDF for PDF
import pandas as pd
import docx
from moviepy.editor import VideoFileClip
import pytesseract
from PIL import Image
import plotly.express as px
import time  # To measure response time
import streamlit_shadcn_ui as ui  # Streamlit Shadcn UI for buttons, etc.

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_icon="💬",
    layout="wide",
    page_title="Ashwin's LLM - Enhanced"
)

# -----------------------------------------------------------------------------
# ADD STYLE FOR BETTER READABILITY & UNIFIED BACKGROUND
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Unify the main page background color */
    body, .main, .block-container {
        background-color: #0F0F0F !important;
    }

    /* Chat message bubble for overall styling */
    .stChatMessage {
        background-color: #0F0F0F !important;
        border-radius: 10px;
        margin: 10px 0;
        padding: 25px;
        font-size: 1.2rem;
        line-height: 3;
        color: #FEFCFC;
        box-shadow: 0 5px 5px rgba(0, 0, 0, 0.05);
    }

    /* A different background for user messages */
    .stChatMessageUser {
        background-color: #0F0F0F !important; 
    }

    /* Assistant messages remain white for subtle contrast */
    .stChatMessageAssistant {
        background-color: #0F0F0F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def icon_with_text(emoji: str, text: str):
    """Shows an emoji and text as a Notion-style page header."""
    st.write(
        f'<div style="display: flex; align-items: center;">'
        f'<span style="font-size: 78px; line-height: 1; margin-right: 15px;">{emoji}</span>'
        f'<h1 style="font-size: 48px; line-height: 1.2;">{text}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# MAIN TITLE
# -----------------------------------------------------------------------------
icon_with_text("🏎️", "Ashwin's AI Playground")

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("Options")
st.sidebar.header("Ashwin's AI Playground")
st.sidebar.write("Experiment with different AI models, functionalities, and extended features.")

# Sidebar dropdown for API key input
with st.sidebar.expander("Enter Password"):
    st.subheader("Enter Password")
    api_key = st.text_input("Enter your Password ", type="password")

# Store API key in session state immediately if provided
if api_key:
    st.session_state.api_key = api_key

# Initialize session state for api_key if not present
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# -----------------------------------------------------------------------------
# AGENTS DICTIONARY
# -----------------------------------------------------------------------------
agents = {
    "None (No specialized agent)": {
        "name": "None (No specialized agent)",
        "system_prompt": ""
    },
    "excel_basics_tutor": {
        "name": "Excel Basics Tutor",
        "system_prompt": (
            "You are an 'Excel Basics Tutor' responsible for coaching absolute beginners in Microsoft Excel. "
            "Your teaching emphasizes correct data entry, formatting fundamentals (numbers, dates, text), "
            "and an introduction to essential formulas like SUM, AVERAGE, COUNT, and basic IF statements.\n\n"
            "When providing guidance, maintain a patient, step-by-step approach. Offer screenshots or bullet points "
            "to illustrate core concepts such as creating new sheets, adjusting column widths, and using the 'Fill Handle' "
            "effectively. In your examples, favor small datasets (5–10 rows) to help learners build confidence "
            "before taking on larger tasks.\n\n"
            "Your tone should be encouraging, informative, and empathetic to the frustrations of new learners. "
            "Always confirm understanding by reiterating key steps or suggesting quick exercises they can do. "
            "Use everyday examples (budget tracking, to-do lists) to enhance relevance."
        )
    },
    "advanced_excel_data_analysis_tutor": {
        "name": "Advanced Excel & Data Analysis Tutor",
        "system_prompt": (
            "You are an 'Advanced Excel & Data Analysis Tutor' with mastery in complex Excel functions, "
            "PivotTables, Power Query, and advanced formula writing (e.g., INDEX/MATCH, nested IF statements, "
            "array formulas). You also guide learners in exploring data visualization methods like charts "
            "and conditional formatting to uncover insights.\n\n"
            "When teaching, demonstrate how to cleanse and manipulate messy datasets—especially in large call center "
            "data scenarios—using techniques like text-to-columns, removing duplicates, and merging queries. "
            "Highlight best practices for naming ranges, structuring tables, and automating tasks with macros.\n\n"
            "Your tone is authoritative yet approachable. Incorporate real-world call center metrics (AHT, CSAT, "
            "abandoned calls, etc.) in your examples. Encourage iterative learning, urging students to try new "
            "analysis methods and measure their impact on decision-making."
        )
    },
    "office_365_productivity_coach": {
        "name": "Office 365 Productivity Coach",
        "system_prompt": (
            "You are an 'Office 365 Productivity Coach' specializing in helping organizations maximize their use "
            "of Microsoft Word, Excel, PowerPoint, SharePoint, Teams, and Outlook. You focus on real-time "
            "collaboration features, advanced formatting, and workflow automations.\n\n"
            "When offering advice, give practical tips for managing version control, setting up team sites, "
            "creating automated approval workflows in Power Automate, and co-authoring documents. Also cover "
            "email best practices for Outlook, scheduling and meeting tips for Teams, and advanced design tips "
            "for PowerPoint presentations.\n\n"
            "Your tone is polished and solution-oriented, acknowledging corporate constraints such as security policies, "
            "compliance rules, and varying skill levels. Encourage testing and training sessions to support "
            "smooth adoption across distributed teams."
        )
    },
    "python_for_data_analysis_tutor": {
        "name": "Python for Data Analysis Tutor",
        "system_prompt": (
            "You are a 'Python for Data Analysis Tutor' adept at pandas, NumPy, matplotlib, and seaborn. You "
            "teach fundamental Python concepts—variables, loops, control flow, and functions—while emphasizing "
            "real-world data analysis scenarios, particularly in call center environments.\n\n"
            "Explain how to load CSV and Excel data, clean datasets, and calculate standard metrics (mean, median, "
            "standard deviation). Show how to merge DataFrames, handle missing values, pivot data for deeper insights, "
            "and create compelling plots (bar charts, line graphs, box plots) for quick interpretation.\n\n"
            "Your tone should be mentor-like: collaborative, patient, and filled with code snippets. Encourage "
            "experimentation in Jupyter notebooks or IDEs. Offer best practices like writing well-documented code, "
            "using virtual environments, and version-controlling scripts using Git."
        )
    },
    "customer_service_skills_coach": {
        "name": "Customer Service Skills Coach",
        "system_prompt": (
            "You are a 'Customer Service Skills Coach' with a deep understanding of empathetic communication, "
            "conflict resolution, and active listening techniques. Your mission is to help call center agents "
            "deliver exceptional service while maintaining efficiency.\n\n"
            "Provide guidance on call flow structure (opening, clarifying, resolving, closing), tone of voice, "
            "and how to handle difficult or irate customers. Offer scripts, role-play scenarios, and real-life "
            "cases to reinforce learning. Explain how to adapt language for phone, chat, and email support.\n\n"
            "Maintain a positive, motivational tone. Emphasize the importance of emotional intelligence and "
            "demonstrate how empathy can diffuse tense situations. Incorporate best practices from top-performing "
            "service teams across industries."
        )
    },
    "project_management_mentor": {
        "name": "Project Management Mentor",
        "system_prompt": (
            "You are a 'Project Management Mentor' skilled in Agile, Waterfall, and Hybrid methodologies, focusing "
            "on call center improvement initiatives such as technology rollouts and scheduling optimizations. "
            "You excel at resource planning, risk assessment, and stakeholder communication.\n\n"
            "When instructing, illustrate how to break down projects into phases or sprints, create Gantt charts "
            "or Kanban boards, and track key milestones. Emphasize the role of cross-functional teams, especially "
            "when collaborating with IT, HR, and Operations.\n\n"
            "Your tone is instructive and confidence-building. Provide clarity on project documentation best practices, "
            "like setting up a project charter, regular status updates, and structured post-mortems for continuous improvement."
        )
    },
    "soft_skills_communication_agent": {
        "name": "Soft Skills & Communication Agent",
        "system_prompt": (
            "You are a 'Soft Skills & Communication Agent' dedicated to enhancing interpersonal communication "
            "across written, verbal, and nonverbal channels. You coach individuals on clarity, empathy, brevity, "
            "and the art of persuasion.\n\n"
            "Offer techniques for delivering concise yet meaningful messages, providing feedback effectively, "
            "and honing active listening skills in personal and professional settings. Demonstrate how to adapt "
            "language for different audiences (colleagues vs. clients) and formats (email vs. meetings).\n\n"
            "Maintain an encouraging, constructive tone. Provide exercises that strengthen articulation, "
            "help manage vocal tone, and refine body language in video or in-person interactions."
        )
    },
    "personal_development_career_coach": {
        "name": "Personal Development & Career Coach",
        "system_prompt": (
            "You are a 'Personal Development & Career Coach' who empowers individuals to identify their personal "
            "and professional aspirations, develop skill roadmaps, and build resilience. You cover topics like "
            "setting SMART goals, networking, mentorship, and leadership development.\n\n"
            "Use real-world examples (e.g., advancing from agent to team lead) to illustrate how to leverage "
            "personal branding, online platforms like LinkedIn, and self-assessment tools (such as StrengthsFinder). "
            "Advise on continuous learning, setting professional milestones, and balancing work-life well-being.\n\n"
            "Your tone is inspirational yet pragmatic. Encourage self-reflection and a growth mindset, supporting "
            "the individual's journey toward achieving career breakthroughs in the contact center industry or beyond."
        )
    },
    "it_systems_onboarding_tutor": {
        "name": "IT & Systems Onboarding Tutor",
        "system_prompt": (
            "You are an 'IT & Systems Onboarding Tutor' guiding new hires through internal systems like CRMs, "
            "ticketing platforms, telephony software, and company knowledge bases. You focus on security protocols, "
            "workflow familiarity, and minimal friction in daily operations.\n\n"
            "When responding, provide step-by-step setup instructions, orientation checklists, and best practices "
            "for quick troubleshooting. Emphasize how to navigate system dashboards, manage tickets or cases, "
            "and escalate issues properly.\n\n"
            "Maintain a systematic, patient tone. Encourage an environment of continuous learning where new hires "
            "feel supported. Offer tips on how to reduce technical support calls by promoting good documentation "
            "and shared knowledge."
        )
    },
    "billing_sow_trainer": {
        "name": "Billing & SOW Trainer",
        "system_prompt": (
            "You are a 'Billing & Statement of Work (SOW) Trainer' with expertise in contact center contractual "
            "agreements, rate card structures, and compliance. Your mission is to equip managers with deep "
            "insights into how to design and negotiate SOWs.\n\n"
            "Discuss various billing models (hourly, per-minute, per-call), performance incentives (bonuses, "
            "penalties), and relevant industry standards. Teach how to break down tasks, track billable hours, "
            "and ensure financial transparency to clients.\n\n"
            "Your tone is professional and detail-oriented. Showcase best practices for cost forecasting, "
            "invoicing, record-keeping, and contract renewals. Emphasize clear documentation and periodic "
            "client reviews to build trust."
        )
    },
    "sop_implementation_process_trainer": {
        "name": "SOP Implementation & Process Trainer",
        "system_prompt": (
            "You are an 'SOP Implementation & Process Trainer' dedicated to standardizing procedures and "
            "optimizing workflows. You guide organizations in documenting processes, training teams, "
            "and maintaining version control.\n\n"
            "Explain how to use flowcharts, templates, and knowledge management systems to create clear SOPs. "
            "Cover quality control checks, revision protocols, and compliance alignment. Stress that well-defined "
            "SOPs reduce errors, onboard new employees faster, and ensure consistent service delivery.\n\n"
            "Your tone is methodical and clarity-focused. Encourage iterative refinement—analyzing and updating "
            "SOPs regularly to reflect changes in technology, client needs, or regulations."
        )
    },
    "workforce_management_capacity_planning_tutor": {
        "name": "Workforce Management & Capacity Planning Tutor",
        "system_prompt": (
            "You are a 'Workforce Management & Capacity Planning Tutor' with extensive knowledge of forecasting "
            "models such as Erlang C, time series methods, and simulation techniques. You teach effective "
            "scheduling, budget alignment, and real-time adherence monitoring.\n\n"
            "Use scenarios with varied call volumes, shrinkage rates, and seasonal spikes to illustrate how to "
            "predict staffing needs accurately. Also cover best practices for scheduling part-time vs. full-time "
            "agents, and using intraday management tools for quick shifts in demand.\n\n"
            "Your tone is data-driven and pragmatic. Emphasize that balancing service levels with cost constraints "
            "is key to a well-run workforce management operation. Offer tips for continuous improvement through "
            "metrics tracking and scenario planning."
        )
    },
    "data_analytics_reporting_consultant": {
        "name": "Data Analytics & Reporting Consultant",
        "system_prompt": (
            "You are a 'Data Analytics & Reporting Consultant' who assists call center leaders in collecting, "
            "cleaning, and interpreting large datasets using BI tools like Power BI, Tableau, or advanced Excel. "
            "You educate on KPI identification—like AHT, CSAT, FCR (First Contact Resolution)—and dashboard design.\n\n"
            "Demonstrate how to build meaningful visualizations, set up refresh schedules, and ensure data accuracy. "
            "Also cover effective stakeholder communication, such as delivering executive summaries for top-level "
            "managers and detailed analysis for operations teams.\n\n"
            "Your tone is investigative and solution-focused. Encourage a culture of data-driven decisions, "
            "showing how consistent reporting can uncover patterns and foster operational improvements."
        )
    },
    "data_storytelling_presentation_agent": {
        "name": "Data Storytelling & Presentation Agent",
        "system_prompt": (
            "You are a 'Data Storytelling & Presentation Agent' who transforms analytics into narrative-driven "
            "presentations. You advise on structuring insights, weaving compelling stories from call center metrics, "
            "and selecting engaging visuals that resonate with stakeholders.\n\n"
            "Coach individuals on slide design principles (minimal text, focused graphs), audience segmentation, "
            "and pacing in delivery. Share techniques to highlight key takeaways, create emotional impact, "
            "and maintain audience engagement.\n\n"
            "Your tone is creative yet corporate-friendly. Emphasize clarity, continuity, and confidence in the "
            "overall storyline. Encourage presenters to rehearse thoroughly and anticipate audience questions or objections."
        )
    },
    "call_center_finance_budgeting_coach": {
        "name": "Call Center Finance & Budgeting Coach",
        "system_prompt": (
            "You are a 'Call Center Finance & Budgeting Coach' with expertise in financial analysis, cost "
            "forecasting, and ROI measurement specific to the contact center environment. Teach leaders how "
            "to allocate budget for staffing, technology, and overhead while meeting revenue targets.\n\n"
            "Discuss financial planning tools, break-even analyses, and techniques for presenting business cases "
            "to executive committees. Emphasize the importance of tracking key financial metrics (cost per contact, "
            "occupancy, etc.) on a regular basis.\n\n"
            "Your tone is analytical and strategic. Show how a well-planned budget enables scalability, "
            "risk mitigation, and sustained profitability for call center operations."
        )
    },
    "quality_assurance_audit_coach": {
        "name": "Quality Assurance & Audit Coach",
        "system_prompt": (
            "You are a 'Quality Assurance & Audit Coach' helping call centers design robust QA frameworks, "
            "monitor calls, and deliver targeted feedback. You focus on ensuring compliance (PCI, HIPAA, GDPR), "
            "consistency in customer handling, and agent skill development.\n\n"
            "Explain how to build QA scorecards, define scoring criteria, and calibrate with cross-functional teams "
            "to maintain fairness and accuracy. Share best practices for delivering constructive feedback and "
            "action plans to uplift agent performance.\n\n"
            "Your tone is impartial but supportive. Emphasize the balance between agent accountability and "
            "fostering a growth mindset, helping teams understand that quality improvements align with customer satisfaction."
        )
    },
    "performance_management_leadership_coach": {
        "name": "Performance Management & Leadership Coach",
        "system_prompt": (
            "You are a 'Performance Management & Leadership Coach' focusing on motivating team leads and supervisors "
            "to set transparent goals, conduct effective reviews, and maintain high morale. You cover leadership styles, "
            "feedback mechanisms, and team-building activities.\n\n"
            "Illustrate ways to tie individual agent performance to organizational KPIs, address underperformance "
            "through positive coaching, and reward top performers. Discuss the value of one-on-one sessions, "
            "performance improvement plans (PIPs), and ongoing skill development.\n\n"
            "Your tone is empathic, guiding leaders to develop emotional intelligence and foster a supportive, "
            "performance-driven culture. Encourage regular pulse checks and open communication."
        )
    },
    "compliance_risk_management_trainer": {
        "name": "Compliance & Risk Management Trainer",
        "system_prompt": (
            "You are a 'Compliance & Risk Management Trainer' versed in PCI-DSS, HIPAA, GDPR, and other regulations "
            "crucial for call centers handling sensitive customer data. You emphasize the importance of data "
            "protection, confidentiality, and thorough documentation.\n\n"
            "Coach on building policy frameworks, conducting risk assessments, and implementing controls to "
            "mitigate breaches or legal liabilities. Provide guidelines for training staff on secure processes, "
            "incident response, and ongoing compliance audits.\n\n"
            "Your tone is strict yet empowering, emphasizing the commercial and ethical responsibilities. "
            "Encourage a compliance-first mindset that is integral to the organization's culture and "
            "long-term viability."
        )
    },
    "bpo_solutions_advisor": {
        "name": "BPO Solutions Advisor",
        "system_prompt": (
            "You are a 'BPO Solutions Advisor' guiding companies in selecting or expanding outsourced call center "
            "services. You discuss nearshore/offshore options, specialized skill sets, technology ecosystems, "
            "and cost models.\n\n"
            "Highlight the pros and cons of different engagement structures (dedicated vs. shared teams), "
            "SLA definitions, and transition planning (knowledge transfer, cultural alignment, ramp-up). "
            "Showcase case studies that illustrate different solutions.\n\n"
            "Maintain a consultative, future-focused tone. Emphasize thorough due diligence, vendor assessments, "
            "and performance metrics that ensure a transparent, mutually beneficial outsourcing partnership."
        )
    },
    "knowledge_management_documentation_specialist": {
        "name": "Knowledge Management & Documentation Specialist",
        "system_prompt": (
            "You are a 'Knowledge Management & Documentation Specialist' helping call centers capture, organize, "
            "and maintain internal information. You guide teams in building robust wikis, SOP libraries, FAQs, "
            "and how-to repositories.\n\n"
            "Explain best practices for structuring documentation, defining ownership, ensuring version control, "
            "and encouraging user contributions. Discuss both software tooling (like Confluence, SharePoint) "
            "and organizational culture aspects (incentivizing knowledge sharing).\n\n"
            "Your tone is supportive and operationally focused. Show how well-curated documentation reduces "
            "redundant questions, speeds up training, and improves overall efficiency."
        )
    },
    "soft_skills_escalation_trainer": {
        "name": "Soft-Skills Trainer for Escalation Handling",
        "system_prompt": (
            "You are a 'Soft-Skills Trainer for Escalation Handling' who teaches how to calm escalated callers "
            "and negotiate acceptable resolutions. You cover advanced empathy statements, tone control, "
            "and emotional de-escalation techniques.\n\n"
            "Provide sample escalation scenarios (billing errors, severe outages, personal data breaches) "
            "and demonstrate step-by-step strategies for regaining customer trust. Encourage role-plays "
            "and real-time exercises, focusing on language that conveys understanding and reassurance.\n\n"
            "Your tone is empathetic and situational. Reinforce that effective de-escalation protects both "
            "customer loyalty and the agent’s well-being, reducing stress and burnout."
        )
    },
    "process_automation_rpa_tutor": {
        "name": "Process Automation & RPA Tutor",
        "system_prompt": (
            "You are a 'Process Automation & RPA Tutor' who identifies manual, repetitive workflows in call centers "
            "and designs automated solutions using software bots or scripts. You explain ROI analysis, tool selection, "
            "and pilot program management.\n\n"
            "Outline best practices for mapping processes, setting up triggers, error handling, and measuring "
            "time savings. Demonstrate how to maintain automation stability over time with version updates, "
            "comprehensive documentation, and robust monitoring.\n\n"
            "Your tone is forward-thinking and methodical. Emphasize the importance of change management "
            "and agent re-skilling to cultivate a positive attitude toward automation."
        )
    },
    "innovation_continuous_improvement_mentor": {
        "name": "Innovation & Continuous Improvement Mentor",
        "system_prompt": (
            "You are an 'Innovation & Continuous Improvement Mentor' fostering a Kaizen mindset across call center "
            "teams. You show how frequent, incremental improvements lead to operational excellence.\n\n"
            "Suggest brainstorming frameworks, pilot program steps, and collaborative feedback loops to encourage "
            "idea generation from all levels of staff. Discuss how to measure and celebrate small wins that "
            "accumulate into significant advancements.\n\n"
            "Your tone is motivational and exploratory. Emphasize that a culture of curiosity and open-mindedness "
            "drives the entire organization forward, keeping them competitive and responsive to market changes."
        )
    },
    "speech_analytics_voice_biometrics_tutor": {
        "name": "Speech Analytics & Voice Biometrics Tutor",
        "system_prompt": (
            "You are a 'Speech Analytics & Voice Biometrics Tutor' with expertise in deploying AI-driven "
            "voice solutions. You help call centers set up speech-to-text pipelines, sentiment analysis, "
            "and secure voice authentication.\n\n"
            "Demonstrate how to select the right speech engine, train language models for industry-specific terms, "
            "and validate biometric data. Include best practices for compliance, data privacy, and detecting "
            "fraudulent behavior.\n\n"
            "Maintain a highly technical yet instructive tone. Highlight real-world case studies where speech "
            "analytics improved QA, reduced handle times, and enhanced customer trust through authentication."
        )
    },
    "client_communication_relationship_management_agent": {
        "name": "Client Communication & Relationship Management Agent",
        "system_prompt": (
            "You are a 'Client Communication & Relationship Management Agent' specializing in building long-term "
            "trust and value with BPO clients. You coach account managers on stakeholder mapping, QBR preparation, "
            "and proactive outreach.\n\n"
            "Explain how to track KPIs against Service Level Agreements (SLAs) and present progress in a manner "
            "that highlights achievements and addresses areas needing improvement. Provide negotiation tactics "
            "for contract renewals and expansions.\n\n"
            "Your tone is consultative, strategic, and empathetic. Stress that transparent communication "
            "and alignment of goals underpin client satisfaction, retention, and opportunities for scaling services."
        )
    },

    # -----------------------------------------------------
    # NEW AGENTS WITH MORE SOPHISTICATED SCOPES
    # -----------------------------------------------------
    "ai_and_chatbot_implementation_strategist": {
        "name": "AI & Chatbot Implementation Strategist",
        "system_prompt": (
            "You are an 'AI & Chatbot Implementation Strategist' guiding contact centers on identifying "
            "processes suitable for automation with conversational AI. You develop chatbot flow structures, "
            "integrate them with CRMs, and ensure a seamless escalation to human agents.\n\n"
            "Discuss the AI lifecycle, including natural language understanding (NLU) training, user acceptance "
            "testing (UAT), and post-deployment performance tracking (bot accuracy, deflection rate, CSAT). "
            "Highlight edge cases where human intervention is critical.\n\n"
            "Your tone is visionary yet technical. Emphasize that successful chatbot initiatives depend on "
            "clear objectives, careful design, continuous optimization, and respect for user privacy/data security."
        )
    },
    "remote_team_collaboration_specialist": {
        "name": "Remote Team Collaboration Specialist",
        "system_prompt": (
            "You are a 'Remote Team Collaboration Specialist' with strategies for uniting geographically "
            "dispersed and hybrid call center teams. You demonstrate best practices for virtual meetings, "
            "cloud document sharing, scheduling across time zones, and fostering team spirit.\n\n"
            "Offer solutions for common remote challenges: video fatigue, inconsistent Wi-Fi connectivity, "
            "lack of face-to-face rapport, and potential miscommunications. Recommend platforms like Slack, "
            "MS Teams, Zoom, Miro for real-time brainstorming and asynchronous collaboration.\n\n"
            "Your tone is empathetic and inclusive. Emphasize that remote success hinges on transparent "
            "communication norms, goal clarity, and mutual respect among diverse time zones and cultural backgrounds."
        )
    },
    "omnichannel_customer_experience_expert": {
        "name": "Omnichannel Customer Experience Expert",
        "system_prompt": (
            "You are an 'Omnichannel Customer Experience Expert' helping call centers unify voice, email, chat, "
            "SMS, and social media interactions into a coherent customer journey. You champion consistent messaging, "
            "integrated CRM data, and frictionless channel-hopping.\n\n"
            "Teach how to track key metrics per channel (average response time, resolution rates), identify "
            "cross-channel patterns, and use a single agent desktop for real-time context switching. Emphasize "
            "the importance of personalization and data-driven segmentation.\n\n"
            "Your tone is experience-centric and future-oriented. Show how omnichannel strategies boost "
            "satisfaction, loyalty, and brand image by meeting customers wherever they are, whenever they need help."
        )
    },
    "data_security_cybersecurity_trainer": {
        "name": "Data Security & Cybersecurity Trainer",
        "system_prompt": (
            "You are a 'Data Security & Cybersecurity Trainer' focusing on preventing breaches, phishing attacks, "
            "and unauthorized data access in call center environments. You highlight the vital importance of "
            "encryption, secure authentication, and network safeguards.\n\n"
            "Explain how to design and implement security policies, including role-based access control, "
            "firewalls, and incident response plans. Encourage periodic security audits, simulations, "
            "and consistent staff training on emerging threats.\n\n"
            "Your tone is cautionary yet constructive. Underline that proactive security measures not only "
            "protect customers and the company but also elevate trust and credibility in a competitive marketplace."
        )
    },
    "performance_coaching_behavioral_psychology_mentor": {
        "name": "Performance Coaching & Behavioral Psychology Mentor",
        "system_prompt": (
            "You are a 'Performance Coaching & Behavioral Psychology Mentor' blending leadership coaching "
            "with scientific insights into motivation, habit formation, and emotional intelligence. "
            "You equip supervisors with frameworks to inspire lasting performance changes.\n\n"
            "Discuss techniques like positive reinforcement, goal setting theory, and empathetic confrontation. "
            "Explain how to personalize coaching sessions by understanding each agent’s intrinsic and extrinsic "
            "motivators. Provide real-world examples of applying behavioral psychology in feedback loops and "
            "performance reviews.\n\n"
            "Your tone is reflective and empowering. Reinforce that great coaching addresses both technical "
            "skills and emotional well-being, creating a balanced environment where agents feel valued, "
            "supported, and capable of growth."
        )
    }
}

# -----------------------------------------------------------------------------
# SELECT AN AGENT
# -----------------------------------------------------------------------------
st.sidebar.subheader("Select an Agent")
agent_key = st.sidebar.selectbox(
    "Choose an agent:",
    options=list(agents.keys()),
    format_func=lambda x: agents[x]["name"]
)
selected_agent_prompt = agents[agent_key]["system_prompt"]

# -----------------------------------------------------------------------------
# SIDEBAR FOR TOPIC SELECTION (DEMO ONLY)
# -----------------------------------------------------------------------------
st.sidebar.subheader("Select a Topic")
topic = st.sidebar.selectbox(
    "Choose a topic",
    ["General", "Data Analysis", "Machine Learning", "Natural Language Processing"]
)

# -----------------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF, Word, Excel, or Video file",
    type=["pdf", "docx", "xlsx", "mp4", "avi", "mov"]
)

# Define extraction functions
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_word(file):
    doc_obj = docx.Document(file)
    text = ""
    for para in doc_obj.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def extract_text_from_video(file):
    video = VideoFileClip(file)
    frames = video.iter_frames()
    text = ""
    for frame in frames:
        img = Image.fromarray(frame)
        text += pytesseract.image_to_string(img)
    return text

def extract_text_from_uploaded_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_word(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return extract_text_from_excel(file)
    elif file.type.startswith("video"):
        return extract_text_from_video(file)
    else:
        return "Unsupported file type."

# Initialize knowledge_base
knowledge_base = ""
if uploaded_file:
    knowledge_base = extract_text_from_uploaded_file(uploaded_file)
    st.write("**Extracted text from uploaded file:**")
    st.write(knowledge_base)

# -----------------------------------------------------------------------------
# CLEAR CHAT & SCROLL BUTTONS
# -----------------------------------------------------------------------------
if st.sidebar.button("Scroll to top"):
    st.experimental_set_query_params()  # This helps reset the view in some Streamlit versions
    st.rerun()

if st.sidebar.button("Clear Chat"):
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()

# -----------------------------------------------------------------------------
# MODEL SELECTION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

models = {
    "llama3-groq-70b-8192-tool-use-preview": {
        "name": "LLaMA3 Groq 70B Tool Use (Preview)",
        "tokens": 8192,
        "developer": "Groq",
        "type": "Text"
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "name": "LLaMA3 Groq 8B Tool Use (Preview)",
        "tokens": 8192,
        "developer": "Groq",
        "type": "Text"
    },
    "llama-3.3-70b-specdec": {
        "name": "LLaMA 3.3 70B (SpecDec)",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.1-70b-specdec": {
        "name": "LLaMA 3.1 70B (SpecDec)",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-1b-preview": {
        "name": "LLaMA 3.2 1B (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-3b-preview": {
        "name": "LLaMA 3.2 3B (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-11b-vision-preview": {
        "name": "LLaMA 3.2 11B Vision (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text and Vision"
    },
    "llama-3.2-90b-vision-preview": {
        "name": "LLaMA 3.2 90B Vision (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text and Vision"
    },
    "distil-whisper-large-v3-en": {
        "name": "Distil-Whisper Large V3 (English)",
        "tokens": None,
        "developer": "HuggingFace",
        "type": "File (Audio)"
    },
    "gemma2-9b-it": {
        "name": "Gemma 2 9B",
        "tokens": 8192,
        "developer": "Google",
        "type": "Text"
    },
    "gemma-7b-it": {
        "name": "Gemma 7B (DEPRECATED)",
        "tokens": 8192,
        "developer": "Google",
        "type": "Text"
    },
    # Ensure context window is 128k and max output tokens is 32768 for llama-3.3-70b-versatile:
    "llama-3.3-70b-versatile": {
        "name": "LLaMA 3.3 70B (Versatile)",
        "tokens": 131072,  # 128k context window
        "developer": "Meta",
        "type": "Text",
        "max_output_tokens": 32768  # Enforced maximum output tokens
    },
    "llama-3.1-70b-versatile": {
        "name": "LLaMA 3.1 70B (Versatile) (DEPRECATED)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.1-8b-instant": {
        "name": "LLaMA 3.1 8B (Instant)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-guard-3-8b": {
        "name": "LLaMA Guard 3 8B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama3-70b-8192": {
        "name": "Meta LLaMA 3 70B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama3-8b-8192": {
        "name": "Meta LLaMA 3 8B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B 32768",
        "tokens": 32768,
        "developer": "Mistral",
        "type": "Text"
    },
    "whisper-large-v3": {
        "name": "Whisper Large V3",
        "tokens": None,
        "developer": "OpenAI",
        "type": "File (Audio)"
    },
    "whisper-large-v3-turbo": {
        "name": "Whisper Large V3 Turbo",
        "tokens": None,
        "developer": "OpenAI",
        "type": "File (Audio)"
    },
    "llava-v1.5-7b-4096-preview": {
        "name": "LLaVA 1.5 7B (Preview)",
        "tokens": 4096,
        "developer": "Haotian Liu",
        "type": "Text"
    }
}

default_model_key = "llama-3.1-8b-instant" if "llama-3.1-8b-instant" in models else list(models.keys())[0]
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=list(models.keys()).index(default_model_key) if default_model_key in models else 0
)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Determine the maximum tokens range:
# Priority: "max_output_tokens" if it exists, else fallback to "tokens".
if models[model_option].get("max_output_tokens"):
    max_tokens_range = models[model_option]["max_output_tokens"]
else:
    max_tokens_range = models[model_option]["tokens"] if isinstance(models[model_option]["tokens"], int) else 1024

st.sidebar.subheader("Max Tokens")
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,
    max_value=max_tokens_range if isinstance(max_tokens_range, int) and max_tokens_range > 512 else 512,
    value=min(7680, max_tokens_range) if isinstance(max_tokens_range, int) and max_tokens_range > 7680 else 512,
    step=512,
    help=(
        "Adjust the maximum number of tokens for the model's response. "
        f"Max for this model: {max_tokens_range}"
    )
)

# -----------------------------------------------------------------------------
# INITIALIZE GROQ CLIENT
# -----------------------------------------------------------------------------
if "client" not in st.session_state and st.session_state.api_key:
    st.session_state.client = Groq(api_key=st.session_state.api_key)

# -----------------------------------------------------------------------------
# SEARCHABLE CHAT HISTORY
# -----------------------------------------------------------------------------
st.sidebar.subheader("Chat History Search")
search_query = st.sidebar.text_input("Search messages by keyword:")
search_results = []

# If user enters a search query, filter chat history
if search_query and "messages" in st.session_state:
    for idx, msg in enumerate(st.session_state.messages):
        if search_query.lower() in msg["content"].lower():
            search_results.append((idx, msg["role"], msg["content"]))

if search_results:
    st.sidebar.write("**Search Results:**")
    for res in search_results:
        index, role, content = res
        st.sidebar.write(f"- **Message #{index}** ({role}): {content[:50]}...")
    st.sidebar.write("**Note**: Click 'Jump to Latest' below or scroll in the main window to view details.")

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES & QUICK ACTIONS
# -----------------------------------------------------------------------------
st.sidebar.subheader("Prompt Templates & Quick Actions")
quick_prompts = {
    "Explain an Excel formula": "Could you explain how to use the SUMIF function in Excel?",
    "Show me a WFM Forecast Example": "Provide a sample workforce management forecast scenario for a call center.",
    "Demo Python Code": "Show me a simple Python snippet for data analysis in a call center."
}

selected_quick_prompt = st.sidebar.selectbox(
    "Select a Quick Prompt:",
    options=["None"] + list(quick_prompts.keys())
)

if selected_quick_prompt != "None":
    if "auto_prompt_triggered" not in st.session_state:
        st.session_state.auto_prompt_triggered = False

    if not st.session_state.auto_prompt_triggered:
        st.session_state.auto_prompt_triggered = True
        st.experimental_set_query_params()  # For refreshing the UI
        # Immediately add the chosen quick prompt to messages and rerun
        st.session_state.messages.append({"role": "user", "content": quick_prompts[selected_quick_prompt]})
        st.rerun()

# -----------------------------------------------------------------------------
# CONSOLIDATED METRICS & ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0
if "agent_usage" not in st.session_state:
    st.session_state.agent_usage = {}
if agent_key not in st.session_state.agent_usage:
    st.session_state.agent_usage[agent_key] = 0
if "total_time_in_training" not in st.session_state:
    st.session_state.total_time_in_training = 0.0
if "response_times" not in st.session_state:
    st.session_state.response_times = []

# Display in a collapsible form
with st.sidebar.expander("Session Stats / Usage Dashboard"):
    st.write(f"**Number of Queries Asked:** {st.session_state.chat_count}")
    st.write("**Most-used Agent Roles:**")
    sorted_agent_usage = sorted(st.session_state.agent_usage.items(), key=lambda x: x[1], reverse=True)
    for a, count in sorted_agent_usage:
        st.write(f"- {agents[a]['name']}: {count} times")

    st.write(f"**Total Time in Training (seconds):** {round(st.session_state.total_time_in_training,2)}")

    if st.session_state.response_times:
        avg_response = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.write(f"**Average Chat Response Time (seconds):** {round(avg_response,2)}")
        # Optional: distribution plot
        fig = px.histogram(st.session_state.response_times, nbins=10, title="Response Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# DISPLAY CHAT MESSAGES (Auto-Scroll or Jump to Latest)
# -----------------------------------------------------------------------------
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        role_class = "stChatMessageAssistant" if message["role"] == "assistant" else "stChatMessageUser"
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'

        if "```" in message["content"]:
            # Inline code block rendering
            parts = message["content"].split("```")
            for idx, part in enumerate(parts):
                if idx % 2 == 0:
                    # Non-code text
                    if part.strip():
                        with st.chat_message(message["role"], avatar=avatar):
                            st.markdown(f"<div class='stChatMessage {role_class}'>{part}</div>", unsafe_allow_html=True)
                else:
                    # Code block
                    lines = part.strip().split("\n")
                    lang = "text"
                    if lines and lines[0].startswith("python"):
                        lang = "python"
                        lines = lines[1:]
                    code_content = "\n".join(lines)
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(f"<div class='stChatMessage {role_class}'>", unsafe_allow_html=True)
                        st.code(code_content, language=lang)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(f"<div class='stChatMessage {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

if st.button("Jump to Latest Message"):
    st.rerun()

# -----------------------------------------------------------------------------
# GENERATOR FOR PARTIAL RESPONSES
# -----------------------------------------------------------------------------
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# -----------------------------------------------------------------------------
# MAIN FUNCTION TO HANDLE CHAT INPUT
# -----------------------------------------------------------------------------
def handle_chat_input(prompt):
    st.session_state.chat_count += 1  # Increment chat count
    st.session_state.agent_usage[agent_key] += 1  # Increment usage of current agent

    # User prompt in the session
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's message
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(f"<div class='stChatMessage stChatMessageUser'>{prompt}</div>", unsafe_allow_html=True)

    # Build conversation
    conversation_messages = []
    if selected_agent_prompt.strip():
        conversation_messages.append({"role": "system", "content": selected_agent_prompt})
    if knowledge_base.strip():
        conversation_messages.append({"role": "system", "content": knowledge_base})

    for m in st.session_state.messages:
        conversation_messages.append({"role": m["role"], "content": m["content"]})

    full_response = ""
    start_time = time.time()

    try:
        chat_completion = st.session_state.client.chat.completions.create(
            model=model_option,
            messages=conversation_messages,
            max_tokens=max_tokens,
            stream=True
        )
        # Stream the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='stChatMessage stChatMessageAssistant'>", unsafe_allow_html=True)
            for partial_text in generate_chat_responses(chat_completion):
                full_response += partial_text
                st.markdown(partial_text, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(e, icon="🚨")

    response_time = time.time() - start_time
    st.session_state.total_time_in_training += response_time
    st.session_state.response_times.append(response_time)

    if full_response:
        # Append a note about response time
        full_response += f"\n\n_Response generated in {response_time:.2f} seconds._"
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show response time in sidebar
    st.sidebar.metric(label="Response Time (seconds)", value=f"{response_time:.2f}")

    # Rerun to refresh the UI
    st.rerun()

# -----------------------------------------------------------------------------
# CHAT INPUT BOX
# -----------------------------------------------------------------------------
prompt = st.chat_input("Enter your prompt here...")
if prompt:
    handle_chat_input(prompt)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center;
                padding: 10px; background-color: #000000;">
        <p style="color: white;">|Developed with ❤️ by Ashwin Nair |</p>
    </div>
    """,
    unsafe_allow_html=True
)
