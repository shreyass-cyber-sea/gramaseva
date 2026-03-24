# Databricks notebook source
# TITLE: GramSeva - Step 4: Gradio UI
# Launch interactive web interface for scheme discovery

# STEP 1: Install required libraries
# MAGIC %pip install sentence-transformers google-generativeai scikit-learn gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# STEP 2: Load Dependencies and Data
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import gradio as gr

print("📊 Loading embeddings data...")

# Verify embeddings table exists
try:
    pdf = spark.table("default.gramseva_embeddings").toPandas().fillna("")
    print(f"✅ Loaded {len(pdf)} schemes from embeddings table")
except Exception as e:
    print("❌ Error: gramseva_embeddings table not found")
    print("Please run gramseva_02_embeddings.py first to generate embeddings")
    raise e

# Load SentenceTransformer model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded")
except Exception as e:
    print("❌ Error loading SentenceTransformer:", str(e))
    raise e

# Convert embeddings to numpy matrix
try:
    embeddings_matrix = np.array([json.loads(e) for e in pdf["embedding"]])
    print(f"✅ Embeddings matrix ready: {embeddings_matrix.shape}")
except Exception as e:
    print("❌ Error processing embeddings:", str(e))
    raise e

# COMMAND ----------

# STEP 3: Configure Gemini API
print("🔧 Configuring Gemini API...")

# Use Databricks secrets for API key security
# Create secret: dbutils.secrets.put("gramseva", "gemini_api_key", "your_api_key")
try:
    api_key = dbutils.secrets.get(scope="gramseva", key="gemini_api_key")
    print("✅ API key loaded from Databricks secrets")
except:
    # Fallback - but user should set up secrets properly
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found. Please set up one of:")
        print("  1. Databricks secret: dbutils.secrets.put('gramseva', 'gemini_api_key', 'your_key')")
        print("  2. Environment variable: GEMINI_API_KEY")
        raise ValueError("Please set up Gemini API key in Databricks secrets or environment variable")
    print("✅ API key loaded from environment variable")

genai.configure(api_key=api_key)
try:
    gemini = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini model configured successfully")
except Exception as e:
    print("❌ Error configuring Gemini model:", str(e))
    raise e

# COMMAND ----------

# STEP 4: Define GramSeva Function
def gramseva(name, occupation, state, income, category, age):
    """
    Main function to find relevant government schemes for a user
    """
    try:
        # Build search query
        query = f"{occupation} {state} government scheme"

        # Generate query embedding
        q_emb = model.encode([query])

        # Find similar schemes using cosine similarity
        scores = cosine_similarity(q_emb, embeddings_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:8]  # Top 8 relevant schemes

        # Build context from top schemes
        context = ""
        for i in top_idx:
            r = pdf.iloc[i]
            context += f"\n- {r['scheme_name']}: Eligibility: {str(r['eligibility'])[:200]} | Benefits: {str(r['benefits'])[:200]}"

        # Create prompt for Gemini
        prompt = f"""You are GramSeva, an AI assistant helping Indian citizens find government schemes they qualify for.

User Profile:
Name: {name}, Occupation: {occupation}, State: {state},
Income: ₹{income}, Category: {category}, Age: {age}

Relevant Government Schemes:
{context}

Based on this user's profile, identify which schemes they are ELIGIBLE for.
For each eligible scheme provide:
1. Scheme name
2. Why they qualify (specific to their profile)
3. Benefit amount in ₹ or concrete terms
4. How to apply (brief steps)

Format as a clean numbered list. Be specific and encouraging.
End with TOTAL estimated annual benefit in ₹."""

        # Get response from Gemini
        response = gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        error_msg = f"""❌ Error processing your request: {str(e)}

Please check:
1. All required fields are filled correctly
2. Internet connection is stable
3. Gemini API key is valid and has quota
4. Try again in a few seconds

If the error persists, contact the administrator."""
        return error_msg

print("✅ GramSeva function ready!")

# COMMAND ----------

# STEP 5: Create and Launch Gradio Interface
print("🚀 Creating Gradio interface...")

try:
    demo = gr.Interface(
        fn=gramseva,
        inputs=[
            gr.Textbox(
                label="Full Name",
                placeholder="e.g. Ramesh Kumar Yadav",
                info="Enter your full name"
            ),
            gr.Dropdown(
                choices=["farmer", "student", "daily wage worker", "small business owner",
                        "unemployed", "woman entrepreneur", "senior citizen", "differently abled"],
                label="Occupation",
                info="Select your primary occupation"
            ),
            gr.Dropdown(
                choices=["Uttar Pradesh", "Maharashtra", "Karnataka", "Bihar", "Rajasthan",
                        "Tamil Nadu", "West Bengal", "Gujarat", "Madhya Pradesh", "Odisha",
                        "Andhra Pradesh", "Telangana", "Kerala", "Punjab", "Haryana"],
                label="State",
                info="Select your state of residence"
            ),
            gr.Number(
                label="Annual Income (₹)",
                value=80000,
                info="Enter your total annual household income in rupees"
            ),
            gr.Dropdown(
                choices=["General", "OBC", "SC", "ST", "EWS"],
                label="Category",
                info="Select your social category"
            ),
            gr.Number(
                label="Age",
                value=30,
                minimum=1,
                maximum=100,
                info="Enter your age in years"
            )
        ],
        outputs=gr.Textbox(
            label="🇮🇳 Your Eligible Government Schemes",
            lines=30,
            info="AI-powered analysis of schemes you qualify for"
        ),
        title="🇮🇳 GramSeva — अपनी योजना खोजें",
        description="""
### Find all Central & State government schemes you qualify for — in seconds!

**Powered by AI + 3,400+ real Indian government schemes**

📋 Simply fill your profile above and discover schemes worth lakhs that you may be missing out on.

🔒 **Privacy**: Your data is processed locally and not stored permanently.
        """,
        examples=[
            ["Ramesh Kumar Yadav", "farmer", "Uttar Pradesh", 80000, "OBC", 42],
            ["Priya Sharma", "student", "Karnataka", 150000, "General", 20],
            ["Sunita Devi", "woman entrepreneur", "Bihar", 60000, "SC", 35],
            ["Arjun Singh", "senior citizen", "Rajasthan", 120000, "General", 68],
            ["Meera Patel", "differently abled", "Gujarat", 40000, "EWS", 28]
        ],
        theme="soft",
        allow_flagging="never"
    )
    print("✅ Gradio interface created successfully")

    # Launch with share=True for public URL in Databricks
    print("🌐 Launching GramSeva app...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
    print("✅ GramSeva app launched successfully!")
    print("📱 Check the output above for the public URL to access your app")

except Exception as e:
    print("❌ Error launching Gradio app:", str(e))
    print("\nTroubleshooting:")
    print("1. Ensure all pip installs completed successfully")
    print("2. Check internet connectivity")
    print("3. Verify Gemini API key is correct")
    print("4. Try restarting the Databricks cluster")
    print("5. Check if port 7860 is available")
    raise e