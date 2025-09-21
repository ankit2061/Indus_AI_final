import streamlit as st
import google.generativeai as genai
import json
import base64
from PIL import Image
import io
import os
from typing import List, Dict
import re

# Configure Streamlit page
st.set_page_config(
    page_title="IndusAI - Artisan Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("AIzaSyCmCa_HDsHsg0oy6H4viaUpoggKPLi0TF4", "")
    if not api_key:
        st.error("Please set GEMINI_API_KEY environment variable or in Streamlit secrets")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

model = init_gemini()

# Seed data for Indian crafts
CRAFTS_DATA = [
    {
        "craft_id": "banaras_brocade",
        "name": "Banaras Brocades and Sarees",
        "region": "Varanasi, Uttar Pradesh",
        "category": "Textile",
        "materials": ["silk", "zari"],
        "techniques": ["brocade weaving"],
        "motifs": ["kalga", "bel", "floral"],
        "gi_tag": True,
        "price_band": "₹10,000–₹150,000",
        "seasonality": ["wedding"],
        "cultural_notes": "Mughal-inspired motifs, heavy zari patterns from third-generation weavers.",
        "story": "On the Varanasi ghats, weaver Rahimbhai sets the jacquard for kalga-bel vines, guiding silk and zari into wedding saris that honor his ustad's Mughal patterns."
    },
    {
        "craft_id": "kashmiri_carpet",
        "name": "Kashmiri Hand-Knotted Carpet",
        "region": "Kashmir",
        "category": "Textile",
        "materials": ["wool", "silk"],
        "techniques": ["hand-knotting"],
        "motifs": ["floral", "medallion"],
        "gi_tag": True,
        "price_band": "₹30,000–₹500,000",
        "seasonality": ["all-season"],
        "cultural_notes": "High knot density carpets with intricate Persian-influenced motifs.",
        "story": "Master weaver Ghulam threads silk warps in his Srinagar workshop, each knot preserving centuries of Kashmiri artistry."
    },
    {
        "craft_id": "bastar_dhokra",
        "name": "Bastar Dhokra",
        "region": "Chhattisgarh",
        "category": "Metal",
        "materials": ["brass", "bronze"],
        "techniques": ["lost-wax casting"],
        "motifs": ["tribal figures", "animals"],
        "gi_tag": True,
        "price_band": "₹1,500–₹80,000",
        "seasonality": ["all-season"],
        "cultural_notes": "27-step process using no moulds, creating unique tribal art pieces.",
        "story": "Tribal artisan Sukhdei shapes beeswax figures of dancing women and elephants before the bronze casting ritual begins."
    },
    {
        "craft_id": "jaipur_blue_pottery",
        "name": "Jaipur Blue Pottery",
        "region": "Rajasthan",
        "category": "Ceramics",
        "materials": ["quartz", "cobalt"],
        "techniques": ["glazing", "painting"],
        "motifs": ["floral", "geometric"],
        "gi_tag": True,
        "price_band": "₹500–₹25,000",
        "seasonality": ["festival"],
        "cultural_notes": "Unique quartz-based pottery with distinctive cobalt blue glazing.",
        "story": "Potter Krishan mixes quartz and cobalt in his Jaipur kiln, creating the signature blue that has defined Rajasthani ceramics for generations."
    },
    {
        "craft_id": "warli_painting",
        "name": "Warli Painting",
        "region": "Maharashtra",
        "category": "Painting",
        "materials": ["rice paste", "gum"],
        "techniques": ["tribal painting"],
        "motifs": ["circles", "triangles", "human figures"],
        "gi_tag": True,
        "price_band": "₹800–₹15,000",
        "seasonality": ["harvest"],
        "cultural_notes": "Ancient tribal art using geometric patterns representing daily life.",
        "story": "Adivasi artist Jivya paints white rice-paste figures on mud walls, celebrating the harvest with circular dance patterns."
    }
]

# Lexicon
LEXICON_RULES = [
    {"contains": ["ikat", "bandha", "sambalpuri"], "add": ["Ikat", "Sambalpuri", "Odisha"]},
    {"contains": ["lost-wax", "dokra", "dhokra", "bastar"], "add": ["Dhokra", "Bastar", "Lost-wax"]},
    {"contains": ["cobalt blue", "quartz", "jaipur", "blue pottery"], "add": ["Blue Pottery", "Jaipur", "Ceramics"]},
    {"contains": ["rice paste", "tribal", "warli"], "add": ["Warli", "Maharashtra", "Tribal Painting"]},
    {"contains": ["brocade", "zari", "varanasi", "banaras"], "add": ["Banaras Brocade", "Varanasi", "Silk"]},
]

ASSESSMENT_RUBRIC = [
    {"axis": "craftsmanship_quality", "description": "Stitch/knot/edge neatness; uniformity; structural integrity"},
    {"axis": "tradition_fidelity", "description": "Authentic materials/techniques; culturally accurate motifs"},
    {"axis": "technique_difficulty", "description": "Complexity of weave/knot/engraving; precision required"},
    {"axis": "finish_durability", "description": "Finishing quality, colorfastness, longevity indicators"},
    {"axis": "originality_within_tradition", "description": "Innovation that respects canonical boundaries"},
    {"axis": "documentation", "description": "Process photos, GI/lineage notes, material sources"}
]

# Mock mentors data
MENTORS = [
    {
        "id": "m1", "name": "Ustad Rahman Sheikh", "crafts": ["Banaras Brocades"],
        "region": "Uttar Pradesh", "languages": ["Hindi", "English"],
        "experience_years": 25, "skills": ["zari work", "loom setup", "traditional motifs"],
        "specialty": "Wedding saree designs with authentic Mughal patterns"
    },
    {
        "id": "m2", "name": "Sukhdei Dhurva", "crafts": ["Bastar Dhokra"],
        "region": "Chhattisgarh", "languages": ["Hindi", "English"],
        "experience_years": 18, "skills": ["lost-wax casting", "tribal motifs", "bronze finishing"],
        "specialty": "Traditional tribal figurines and ceremonial items"
    },
    {
        "id": "m3", "name": "Krishan Sharma", "crafts": ["Jaipur Blue Pottery"],
        "region": "Rajasthan", "languages": ["Hindi", "English"],
        "experience_years": 15, "skills": ["glazing", "kiln management", "cobalt techniques"],
        "specialty": "Contemporary adaptations of traditional blue pottery"
    },
    {
        "id": "m4", "name": "Jivya Soma Mashe", "crafts": ["Warli Painting"],
        "region": "Maharashtra", "languages": ["Marathi", "Hindi", "English"],
        "experience_years": 30, "skills": ["traditional motifs", "natural pigments", "wall painting"],
        "specialty": "Authentic tribal storytelling through geometric art"
    }
]

# Session state
if 'crafts_data' not in st.session_state:
    st.session_state.crafts_data = {craft['craft_id']: craft for craft in CRAFTS_DATA}

if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []

if 'price_estimates' not in st.session_state:
    st.session_state.price_estimates = []

# Functions
def parse_price_band(band: str):
    """Extract low and high prices from band string"""
    nums = re.findall(r'[\d,]+', band)
    if len(nums) >= 2:
        low = int(nums[0].replace(',', ''))
        high = int(nums[-1].replace(',', ''))
        return low, high
    return 1000, 100000

def auto_tag_text(text: str):
    """Auto-tag text based on lexicon rules"""
    text_lower = text.lower()
    tags = set()

    for rule in LEXICON_RULES:
        if any(keyword in text_lower for keyword in rule["contains"]):
            tags.update(rule["add"])

    return list(tags)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def match_mentors(craft_name: str, target_skills: List[str], region: str,
                  language_prefs: List[str], experience_level: str, weak_axes: List[str]):
    ranked = []

    for mentor in MENTORS:
        score = 0

        # Craft match
        if any(craft.lower() in craft_name.lower() or craft_name.lower() in craft.lower()
               for craft in mentor["crafts"]):
            score += 4

        # Skills overlap
        mentor_skills_lower = [skill.lower() for skill in mentor["skills"]]
        skill_overlap = sum(1 for skill in target_skills
                            if any(skill.lower() in ms for ms in mentor_skills_lower))
        score += min(skill_overlap, 3)

        # Region proximity
        if region.lower() in mentor["region"].lower():
            score += 2

        # Language preference
        mentor_langs_lower = [lang.lower() for lang in mentor["languages"]]
        lang_overlap = sum(1 for lang in language_prefs
                           if lang.lower() in mentor_langs_lower)
        score += min(lang_overlap, 2)

        # Experience matching
        if experience_level in ["beginner", "intermediate"] and mentor["experience_years"] >= 10:
            score += 1

        # Weak axes specific matching
        for weak_axis in weak_axes:
            if "finish" in weak_axis.lower() and "finishing" in mentor["skills"]:
                score += 1
            if "traditional" in weak_axis.lower() and "traditional" in " ".join(mentor["skills"]).lower():
                score += 1

        ranked.append((score, mentor))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:3]

# Main UI
st.title("🎨 IndusAI - Traditional Craft Assistant")
st.markdown("*Empowering Indian artisans with AI-driven market insights and skill development*")

# Sidebar
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

# API Key input in sidebar
if not os.getenv("GEMINI_API_KEY"):
    api_key_input = st.sidebar.text_input("Enter Gemini API Key:", type="password",
                                          help="Get your API key from Google AI Studio")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.sidebar.success("API Key set successfully!")
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["💰 Price Estimator", "🔍 Cultural Explorer",
                                  "🏷️ Auto-Tagging & Moderation", "📊 Skill Assessment"])

# Tab 1: Price Estimator
with tab1:
    st.header("Market Price Estimator")
    st.markdown("Get AI-powered price estimates for your traditional crafts based on materials, techniques, and market factors.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Craft Details")
        selected_craft = st.selectbox("Select Craft Type:",
                                      options=list(st.session_state.crafts_data.keys()),
                                      format_func=lambda x: st.session_state.crafts_data[x]["name"])

        size_cm = st.number_input("Size (cm):", min_value=1.0, value=50.0, step=1.0)
        hours = st.number_input("Hours of work:", min_value=1.0, value=20.0, step=1.0)
        materials_detail = st.text_area("Material Details:",
                                        placeholder="e.g., Pure silk, 24k gold zari, natural dyes...")

        embellishment_level = st.selectbox("Embellishment Level:",
                                           ["none", "low", "medium", "high"])
        seasonal = st.checkbox("Seasonal/Festival item")

    with col2:
        st.subheader("Market Context")
        seller_region = st.text_input("Seller Region:", value="Mumbai, Maharashtra")
        demand_note = st.text_area("Market Demand Context:",
                                   placeholder="e.g., Wedding season, trending on social media...")
        condition = st.selectbox("Condition:", ["new", "vintage", "restored"])

        if st.button("🔮 Estimate Price", type="primary"):
            with st.spinner("Analyzing market data and craft details..."):
                craft_info = st.session_state.crafts_data[selected_craft]
                base_low, base_high = parse_price_band(craft_info["price_band"])

                # Apply factors
                size_factor = 1.0 + min(size_cm / 100.0, 3.0) * 0.2
                time_factor = 1.0 + min(hours / 40.0, 2.0) * 0.3
                emb_factors = {"none": 1.0, "low": 1.05, "medium": 1.15, "high": 1.3}
                emb_factor = emb_factors.get(embellishment_level, 1.0)
                season_factor = 1.15 if seasonal else 1.0

                anchor_low = base_low * size_factor * time_factor * emb_factor * season_factor
                anchor_high = base_high * size_factor * time_factor * emb_factor * season_factor
                anchor_mid = (anchor_low + anchor_high) / 2

                prompt = f"""
You are an expert Indian craft pricing analyst. Analyze this craft and provide a realistic price estimate in JSON format.

Craft: {craft_info['name']} from {craft_info['region']}
Base Materials: {craft_info['materials']}
Techniques: {craft_info['techniques']}
GI Tag Status: {craft_info['gi_tag']}

Item Details:
- Size: {size_cm}cm
- Work Hours: {hours}
- Materials: {materials_detail}
- Embellishment: {embellishment_level}
- Seasonal: {seasonal}
- Region: {seller_region}
- Market Context: {demand_note}
- Condition: {condition}

Reference Price Band: {craft_info['price_band']}
Calculated Anchors: Low ≈₹{int(anchor_low)}, Mid ≈₹{int(anchor_mid)}, High ≈₹{int(anchor_high)}

Return ONLY a JSON object with:
{{
  "low": <integer>,
  "mid": <integer>, 
  "high": <integer>,
  "rationale": ["<bullet point 1>", "<bullet point 2>", "<bullet point 3>"],
  "risk_flags": ["<any concerns>"],
  "confidence": "<high/medium/low>"
}}

Consider Indian market dynamics, regional variations, and cultural significance.
"""

                try:
                    response = model.generate_content(prompt)
                    result = response.text

                    # Try to extract JSON from response
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        estimate = json.loads(json_str)

                        # Store in session state
                        estimate_record = {
                            "craft": craft_info["name"],
                            "estimate": estimate,
                            "details": {
                                "size": size_cm, "hours": hours,
                                "materials": materials_detail,
                                "embellishment": embellishment_level
                            }
                        }
                        st.session_state.price_estimates.append(estimate_record)

                        # Display results
                        st.success("Price Analysis Complete!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Low Estimate", f"₹{estimate['low']:,}")
                        with col2:
                            st.metric("Mid Estimate", f"₹{estimate['mid']:,}")
                        with col3:
                            st.metric("High Estimate", f"₹{estimate['high']:,}")

                        st.subheader("Analysis Rationale")
                        for point in estimate['rationale']:
                            st.write(f"• {point}")

                        if estimate.get('risk_flags'):
                            st.warning("⚠️ Risk Flags")
                            for flag in estimate['risk_flags']:
                                st.write(f"• {flag}")

                        st.info(f"Confidence Level: {estimate['confidence'].title()}")

                    else:
                        st.error("Could not parse price estimate. Please try again.")
                        st.text(result)

                except Exception as e:
                    st.error(f"Error generating estimate: {str(e)}")

    # Price history
    if st.session_state.price_estimates:
        st.subheader("Recent Estimates")
        for i, estimate in enumerate(reversed(st.session_state.price_estimates[-3:])):
            with st.expander(f"{estimate['craft']} - ₹{estimate['estimate']['mid']:,}"):
                st.write(f"**Size:** {estimate['details']['size']}cm")
                st.write(f"**Hours:** {estimate['details']['hours']}")
                st.write(f"**Range:** ₹{estimate['estimate']['low']:,} - ₹{estimate['estimate']['high']:,}")

# Tab 2: Cultural Explorer
with tab2:
    st.header("Cultural Heritage Explorer")
    st.markdown("Discover the rich heritage and stories behind traditional Indian crafts.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Filters")
        filter_category = st.selectbox("Category:",
                                       ["All"] + list(set(craft["category"] for craft in CRAFTS_DATA)))
        filter_region = st.text_input("Region (contains):")
        filter_gi = st.checkbox("GI Tagged Only")
        search_query = st.text_input("Search:", placeholder="Enter keywords...")

    with col2:
        st.subheader("Craft Collection")

        # Filter crafts
        filtered_crafts = []
        for craft in CRAFTS_DATA:
            if filter_category != "All" and craft["category"] != filter_category:
                continue
            if filter_region and filter_region.lower() not in craft["region"].lower():
                continue
            if filter_gi and not craft["gi_tag"]:
                continue
            if search_query:
                searchable = f"{craft['name']} {craft['region']} {' '.join(craft['materials'])} {' '.join(craft['techniques'])}"
                if search_query.lower() not in searchable.lower():
                    continue
            filtered_crafts.append(craft)

        for craft in filtered_crafts:
            with st.expander(f"🎨 {craft['name']}", expanded=False):
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.write(f"**Region:** {craft['region']}")
                    st.write(f"**Category:** {craft['category']}")
                    st.write(f"**Materials:** {', '.join(craft['materials'])}")
                    st.write(f"**Techniques:** {', '.join(craft['techniques'])}")
                    if craft['gi_tag']:
                        st.success("🏆 GI Tagged")

                with col_right:
                    st.write(f"**Motifs:** {', '.join(craft['motifs'])}")
                    st.write(f"**Price Range:** {craft['price_band']}")
                    st.write(f"**Seasonality:** {', '.join(craft['seasonality'])}")

                st.markdown("**Cultural Heritage:**")
                st.write(craft['cultural_notes'])

                if craft.get('story'):
                    st.markdown("**Artisan Story:**")
                    st.write(f"*{craft['story']}*")

# Tab 3: Auto-Tagging & Moderation
with tab3:
    st.header("Content Tagging & Cultural Moderation")
    st.markdown("Automatically tag craft descriptions and ensure cultural authenticity.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Content Analysis")
        title = st.text_input("Craft Title:")
        description = st.text_area("Description:", height=150,
                                   placeholder="Describe your craft, materials, techniques, and cultural significance...")

        if st.button("🏷️ Analyze & Tag", type="primary"):
            if title or description:
                # Auto-tagging
                text_content = f"{title} {description}"
                auto_tags = auto_tag_text(text_content)

                # Get AI suggestions
                with st.spinner("Analyzing content for cultural accuracy..."):
                    tag_prompt = f"""
Analyze this Indian craft content and suggest appropriate metadata tags.
Title: {title}
Description: {description}

Suggest 5-8 relevant tags covering:
- Craft type/technique
- Region/state
- Materials
- Motifs/patterns
- Cultural context

Return only a JSON array of tags: ["tag1", "tag2", ...]
"""

                    moderation_prompt = f"""
Review this Indian craft listing for cultural and heritage concerns:

Title: {title}
Description: {description}

Check for:
- GI tag misuse or counterfeit claims
- Incorrect region/community attribution  
- Sacred motif misrepresentation
- Culturally insensitive descriptions
- Historical inaccuracies

Return JSON with:
{{
  "flags": ["<specific concerns>"],
  "severity": <1-5>,
  "guidance": ["<suggestion 1>", "<suggestion 2>"],
  "overall_status": "<approved/review_needed/rejected>"
}}
"""

                    try:
                        # Get AI tags
                        tag_response = model.generate_content(tag_prompt)
                        tag_result = tag_response.text

                        # Get moderation analysis
                        mod_response = model.generate_content(moderation_prompt)
                        mod_result = mod_response.text

                        st.session_state.last_analysis = {
                            "auto_tags": auto_tags,
                            "ai_tags": tag_result,
                            "moderation": mod_result
                        }

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")

    with col2:
        st.subheader("Analysis Results")

        if hasattr(st.session_state, 'last_analysis'):
            analysis = st.session_state.last_analysis

            # Auto-detected tags
            if analysis['auto_tags']:
                st.write("**Rule-based Tags:**")
                for tag in analysis['auto_tags']:
                    st.code(tag, language=None)

            # AI suggested tags
            st.write("**AI Suggested Tags:**")
            st.code(analysis['ai_tags'], language="json")

            # Moderation results
            st.write("**Cultural Moderation:**")
            try:
                # Try to parse JSON from moderation response
                mod_text = analysis['moderation']
                json_start = mod_text.find('{')
                json_end = mod_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    mod_json = json.loads(mod_text[json_start:json_end])

                    status = mod_json.get('overall_status', 'unknown')
                    if status == 'approved':
                        st.success("✅ Content Approved")
                    elif status == 'review_needed':
                        st.warning("⚠️ Review Needed")
                    else:
                        st.error("❌ Content Rejected")

                    if mod_json.get('flags'):
                        st.write("**Concerns:**")
                        for flag in mod_json['flags']:
                            st.write(f"• {flag}")

                    if mod_json.get('guidance'):
                        st.write("**Guidance:**")
                        for guide in mod_json['guidance']:
                            st.write(f"• {guide}")

                    st.write(f"**Severity Level:** {mod_json.get('severity', 'N/A')}/5")
                else:
                    st.code(mod_text, language="json")

            except:
                st.code(analysis['moderation'], language="json")

# Tab 4: Skill Assessment
with tab4:
    st.header("Skill Assessment & Mentorship Matching")
    st.markdown("Upload images of your work for AI assessment and get matched with expert mentors.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Craft Assessment")
        craft_name = st.selectbox("Your Craft:",
                                  options=[craft["name"] for craft in CRAFTS_DATA])

        uploaded_files = st.file_uploader("Upload craft images:",
                                          type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)

        assessment_notes = st.text_area("Additional Notes:",
                                        placeholder="Describe your work process, challenges, or specific areas you'd like feedback on...")

        if uploaded_files and st.button("📊 Assess My Work", type="primary"):
            with st.spinner("Analyzing your craft work..."):
                # Display uploaded images
                st.write("**Uploaded Images:**")
                cols = st.columns(min(len(uploaded_files), 3))
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    with cols[i % 3]:
                        st.image(image, caption=f"Image {i + 1}", use_container_width=True)

                # Prepare rubric text
                rubric_text = "\n".join([f"- {r['axis']}: {r['description']}" for r in ASSESSMENT_RUBRIC])

                assessment_prompt = f"""
You are an expert Indian craft assessor. Analyze the uploaded craft work.

Craft Type: {craft_name}
Artist Notes: {assessment_notes}

Assessment Rubric (score each 1-5):
{rubric_text}

Based on the images and notes, provide:
{{
  "scores": {{
    "craftsmanship_quality": <1-5>,
    "tradition_fidelity": <1-5>,
    "technique_difficulty": <1-5>,
    "finish_durability": <1-5>, 
    "originality_within_tradition": <1-5>,
    "documentation": <1-5>
  }},
  "overall_score": <1-5>,
  "strengths": ["<strength 1>", "<strength 2>"],
  "areas_for_improvement": ["<area 1>", "<area 2>"],
  "specific_feedback": ["<detailed feedback>"],
  "weak_axes": ["<axis names that scored <3>"]
}}

Focus on constructive feedback that respects traditional techniques while encouraging growth.
"""

                try:
                    # Process first image for assessment
                    if uploaded_files:
                        first_image = Image.open(uploaded_files[0])
                        response = model.generate_content([assessment_prompt, first_image])
                        result = response.text

                        # Parse assessment result
                        json_start = result.find('{')
                        json_end = result.rfind('}') + 1

                        if json_start != -1 and json_end > json_start:
                            assessment = json.loads(result[json_start:json_end])

                            # Store in session history
                            assessment_record = {
                                "craft": craft_name,
                                "assessment": assessment,
                                "notes": assessment_notes
                            }
                            st.session_state.assessment_history.append(assessment_record)

                            st.session_state.current_assessment = assessment
                            st.success("Assessment Complete!")
                        else:
                            st.error("Could not parse assessment. Please try again.")
                            st.text(result)

                except Exception as e:
                    st.error(f"Assessment error: {str(e)}")

    with col2:
        st.subheader("Assessment Results")

        if hasattr(st.session_state, 'current_assessment'):
            assessment = st.session_state.current_assessment

            # Overall score
            overall = assessment.get('overall_score', 0)
            st.metric("Overall Score", f"{overall}/5",
                      delta=None, delta_color="normal")

            # Individual scores
            st.write("**Detailed Scores:**")
            scores = assessment.get('scores', {})
            for axis, score in scores.items():
                axis_name = axis.replace('_', ' ').title()
                st.progress(score / 5, text=f"{axis_name}: {score}/5")

            # Strengths
            if assessment.get('strengths'):
                st.write("**Strengths:**")
                for strength in assessment['strengths']:
                    st.success(f"✅ {strength}")

            # Areas for improvement
            if assessment.get('areas_for_improvement'):
                st.write("**Areas for Improvement:**")
                for area in assessment['areas_for_improvement']:
                    st.warning(f"📈 {area}")

            # Specific feedback
            if assessment.get('specific_feedback'):
                st.write("**Detailed Feedback:**")
                for feedback in assessment['specific_feedback']:
                    st.info(feedback)

    # Mentorship Matching Section
    if hasattr(st.session_state, 'current_assessment'):
        st.markdown("---")
        st.subheader("🤝 Mentorship Matching")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Find Your Ideal Mentor**")
            target_skills = st.multiselect("Skills you want to learn:",
                                           ["traditional motifs", "finishing", "zari work",
                                            "natural pigments", "loom setup", "glazing",
                                            "lost-wax casting", "tribal motifs"])

            preferred_region = st.text_input("Preferred Region:", value="Any")
            languages = st.multiselect("Language Preferences:",
                                       ["Hindi", "English", "Marathi", "Bengali", "Tamil"])
            experience_level = st.selectbox("Your Level:",
                                            ["beginner", "intermediate", "advanced"])

            if st.button("🔍 Find Mentors"):
                weak_axes = st.session_state.current_assessment.get('weak_axes', [])
                matches = match_mentors(craft_name, target_skills, preferred_region,
                                        languages, experience_level, weak_axes)
                st.session_state.mentor_matches = matches

        with col2:
            if hasattr(st.session_state, 'mentor_matches'):
                st.write("**Recommended Mentors:**")

                for score, mentor in st.session_state.mentor_matches:
                    with st.expander(f"🎨 {mentor['name']} (Match: {score}/10)"):
                        st.write(f"**Specializes in:** {', '.join(mentor['crafts'])}")
                        st.write(f"**Region:** {mentor['region']}")
                        st.write(f"**Experience:** {mentor['experience_years']} years")
                        st.write(f"**Languages:** {', '.join(mentor['languages'])}")
                        st.write(f"**Skills:** {', '.join(mentor['skills'])}")
                        st.write(f"**Specialty:** {mentor['specialty']}")

                        if st.button(f"Connect with {mentor['name']}", key=f"connect_{mentor['id']}"):
                            st.success(f"Connection request sent to {mentor['name']}! 🎉")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎨 <strong>IndusAI</strong> - Preserving India's craft heritage through AI innovation</p>
    <p><em>Built with ❤️ for Indian artisans</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use IndusAI:**

1. **Price Estimator**: Get market value estimates for your crafts
2. **Cultural Explorer**: Browse traditional crafts and their stories  
3. **Auto-Tagging**: Analyze and tag craft descriptions
4. **Skill Assessment**: Upload images for AI feedback and mentor matching

**Need help?** Each tab includes detailed guidance.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Craft Statistics:**")
st.sidebar.metric("Total Crafts", len(CRAFTS_DATA))
st.sidebar.metric("GI Tagged", sum(1 for c in CRAFTS_DATA if c['gi_tag']))
st.sidebar.metric("Regions Covered", len(set(c['region'] for c in CRAFTS_DATA)))