# Databricks notebook source
# MAGIC %pip install sentence-transformers google-generativeai scikit-learn gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Data and Initialize Models
from sentence_transformers import SentenceTransformer
import numpy as np, json
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import gradio as gr

# Verify embeddings table exists
try:
    pdf = spark.table("default.gramseva_embeddings").toPandas().fillna("")
    print(f"✅ Loaded {len(pdf)} schemes from embeddings table")
except Exception as e:
    print("❌ Error: gramseva_embeddings table not found")
    print("Please run Notebook 2.py first to generate embeddings")
    raise e

# Load model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded")
except Exception as e:
    print("❌ Error loading SentenceTransformer:", str(e))
    raise e

# Convert embeddings
try:
    embeddings_matrix = np.array([json.loads(e) for e in pdf["embedding"]])
    print(f"✅ Embeddings matrix ready: {embeddings_matrix.shape}")
except Exception as e:
    print("❌ Error processing embeddings:", str(e))
    raise e

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

def gramseva(name, occupation, state, income, category, age):
    try:
        query = f"{occupation} {state} government scheme"
        q_emb = model.encode([query])
        scores = cosine_similarity(q_emb, embeddings_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:8]

        context = ""
        for i in top_idx:
            r = pdf.iloc[i]
            context += f"\n- {r['scheme_name']}: Eligibility: {str(r['eligibility'])[:200]} | Benefits: {str(r['benefits'])[:200]}"

        prompt = f"""You are GramSeva. Find schemes for:
Name: {name}, Occupation: {occupation}, State: {state},
Income: ₹{income}, Category: {category}, Age: {age}

Schemes:
{context}

List eligible schemes with name, why they qualify, benefit amount, how to apply.
End with TOTAL estimated annual benefit in ₹."""

        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = f"❌ Error processing request: {str(e)}\n\nPlease check:\n1. All required fields are filled\n2. Internet connection for Gemini API\n3. API key is valid and has quota"
        return error_msg

try:
    demo = gr.Interface(
        fn=gramseva,
        inputs=[
            gr.Textbox(label="Full Name", placeholder="e.g. Ramesh Yadav"),
            gr.Dropdown(["farmer", "student", "daily wage worker", "small business owner", "unemployed", "woman entrepreneur"], label="Occupation"),
            gr.Dropdown(["Uttar Pradesh", "Maharashtra", "Karnataka", "Bihar", "Rajasthan", "Tamil Nadu", "West Bengal", "Gujarat", "Madhya Pradesh"], label="State"),
            gr.Number(label="Annual Income (₹)", value=80000),
            gr.Dropdown(["General", "OBC", "SC", "ST"], label="Category"),
            gr.Number(label="Age", value=30)
        ],
        outputs=gr.Textbox(label="🇮🇳 Your Eligible Government Schemes", lines=25),
        title="🇮🇳 GramSeva — Apni Yojana Khojein",
        description="### Find all Central & State government schemes you qualify for — in seconds.\nPowered by AI + 3,400 real government schemes.",
        examples=[
            ["Ramesh Yadav", "farmer", "Uttar Pradesh", 80000, "OBC", 42],
            ["Priya Sharma", "student", "Karnataka", 150000, "General", 20],
            ["Sunita Devi", "woman entrepreneur", "Bihar", 60000, "SC", 35]
        ]
    )
    print("✅ Gradio interface created successfully")

    # Launch with share=True for public URL in Databricks
    demo.launch(share=True)
    print("✅ GramSeva app launched! Check the output above for the public URL.")

except Exception as e:
    print("❌ Error launching Gradio app:", str(e))
    print("This may be due to:")
    print("1. Missing dependencies - ensure all pip installs completed")
    print("2. Network issues - check internet connectivity")
    print("3. Gradio version compatibility - try restarting the cluster")
    raise e

# COMMAND ----------

