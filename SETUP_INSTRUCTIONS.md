# 🇮🇳 GramSeva - Databricks Setup Instructions

## 🚀 Quick Start Guide

This guide will help you run the GramSeva application in Databricks without errors.

### Prerequisites
✅ Databricks workspace (Community Edition works)
✅ Gemini API key from [Google AI Studio](https://aistudio.google.com)
✅ Internet connection for Kaggle dataset download

---

## 📋 Step-by-Step Setup

### Step 1: Setup Gemini API Key (CRITICAL - Security Fix)
**Choose ONE of these methods:**

#### Method A: Databricks Secrets (Recommended)
```python
# In any Databricks cell, run once:
dbutils.secrets.put(scope="gramseva", key="gemini_api_key", value="YOUR_API_KEY_HERE")
```

#### Method B: Environment Variable
```bash
# Set in your cluster environment variables:
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 2: Run Notebooks in Order
Execute these notebooks **sequentially**:

1. **`gramseva_01_setup.py`** - Auto-download dataset from Kaggle → Delta Lake tables
2. **`gramseva_02_embeddings.py`** - Generate text embeddings
3. **`gramseva_03_rag.py`** - Test RAG pipeline (optional)
4. **`gramseva_04_ui.py`** - Launch Gradio web interface

---

## 🔧 What We Fixed

### 🚨 Security Issues Fixed:
- ❌ **Before**: Hardcoded API keys exposed in code
- ✅ **After**: Secure Databricks secrets or environment variables

### 🐛 Runtime Errors Fixed:
- ❌ **Before**: Missing table dependencies, undefined variables
- ✅ **After**: Proper error handling and validation

### 📝 Code Quality Improvements:
- ✅ **Automated Kaggle dataset download** - no manual CSV upload needed
- ✅ Consistent file naming (`gramseva_01_*.py`)
- ✅ Better error messages and troubleshooting
- ✅ Proper library installation with restarts
- ✅ Gradio launch with share URLs for Databricks

---

## 🏃‍♂️ Running the App

After completing setup:

1. **Run** `gramseva_04_ui.py`
2. **Look for** the Gradio public URL in the output
3. **Click** the link to open GramSeva
4. **Test** with example profiles:
   - Ramesh (Farmer, UP, ₹80k, OBC, 42 years)
   - Priya (Student, Karnataka, ₹1.5L, General, 20 years)

---

## 🐛 Troubleshooting

### Common Issues:

**❌ "gramseva_schemes table not found"**
- **Solution**: Run `gramseva_01_setup.py` first

**❌ "Dataset download failed" or "No CSV file found"**
- **Solution**: Check internet connection and Kaggle access; dataset auto-downloads from Kaggle

**❌ "API key not found"**
- **Solution**: Set up Gemini API key using secrets or environment variable

**❌ "Module not found" errors**
- **Solution**: Restart cluster and re-run pip installs

**❌ Gradio not launching**
- **Solution**: Check internet connection, try restarting cluster

### Getting Help:
- Check error messages in notebook outputs
- Verify all prerequisites are met
- Try running notebooks one cell at a time
- Restart Databricks cluster if needed

---

## 🎯 Expected Results

A successful run will show:
- ✅ All tables created with row counts
- ✅ Embeddings generated (3,400+ schemes)
- ✅ Gradio app launched with public URL
- ✅ Working web interface for scheme discovery

**Sample Output for a Farmer:**
```
🇮🇳 Your Eligible Government Schemes:

1. PM-KISAN Samman Nidhi
   - Why you qualify: Small/marginal farmer with land records
   - Benefits: ₹6,000 annually in 3 instalments
   - Application: Visit nearest CSC or PM-KISAN portal

2. Pradhan Mantri Fasal Bima Yojana
   - Why you qualify: Farmer engaged in agriculture
   - Benefits: Crop insurance up to ₹2,00,000
   - Application: Through banks or insurance companies

TOTAL ESTIMATED ANNUAL BENEFIT: ₹8,00,000+
```

---

## 📚 Technical Details

- **Platform**: Databricks (Spark + MLflow)
- **Data**: 3,400 government schemes in Delta Lake
- **AI**: Sentence-BERT embeddings + Gemini 1.5 Flash
- **UI**: Gradio with public sharing
- **Search**: Cosine similarity-based retrieval

---

**🎉 That's it! You now have a fully working GramSeva deployment that helps citizens find government schemes they qualify for.**