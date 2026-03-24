# Databricks notebook source
# TITLE: GramSeva - Step 2: Generate Embeddings
# Create text embeddings for semantic search

# STEP 1: Verify prerequisites
try:
    df = spark.table("default.gramseva_schemes")
    print(f"✅ Found gramseva_schemes table with {df.count()} rows")
except Exception as e:
    print("❌ Error: gramseva_schemes table not found")
    print("Please run gramseva_01_setup.py first to create the base tables")
    raise e

# COMMAND ----------

# STEP 2: Install required libraries
# MAGIC %pip install sentence-transformers
# After installing packages, restart Python to avoid import issues
dbutils.library.restartPython()

# COMMAND ----------

# STEP 3: Clean and prepare data for embeddings
from pyspark.sql.functions import col, concat_ws, lower, trim

print("🧹 Cleaning and preparing data...")

# Combine key fields into one searchable text column for embeddings
df_clean = df.select(
    col("scheme_name"),
    col("schemeCategory").alias("category"),
    col("level"),           # Central / State
    col("details"),
    col("eligibility"),
    col("benefits"),
    col("application"),
    col("documents")
).dropna(subset=["scheme_name", "eligibility"])

# Create a combined text field for embedding
df_clean = df_clean.withColumn(
    "combined_text",
    concat_ws(" | ",
        col("scheme_name"),
        col("category"),
        col("eligibility"),
        col("benefits")
    )
)

print(f"✅ Clean schemes: {df_clean.count()}")

# Save cleaned version to Delta
df_clean.write.format("delta").mode("overwrite").saveAsTable("default.gramseva_clean")
print("✅ gramseva_clean table saved!")

# COMMAND ----------

# STEP 4: Generate embeddings using SentenceTransformers
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# Load model (lightweight, fast)
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded successfully")
except Exception as e:
    print("❌ Error loading SentenceTransformer model:", str(e))
    print("This may be due to internet connectivity or model download issues")
    raise e

# Pull cleaned data to Pandas for embedding (3400 rows is manageable)
pdf = df_clean.toPandas()
pdf = pdf.fillna("")

print(f"🔄 Generating embeddings for {len(pdf)} schemes...")

# Generate embeddings in batches
try:
    embeddings = model.encode(
        pdf["combined_text"].tolist(),
        batch_size=32,  # Smaller batch size for stability
        show_progress_bar=True
    )
    print("✅ Embeddings generated successfully!")
except Exception as e:
    print("❌ Error generating embeddings:", str(e))
    raise e

# COMMAND ----------

# STEP 5: Store embeddings back to Delta Lake
print("💾 Storing embeddings to Delta Lake...")

# Store embeddings as JSON strings (Delta-compatible)
pdf["embedding"] = [json.dumps(e.tolist()) for e in embeddings]

# Convert back to Spark and save
df_with_embeddings = spark.createDataFrame(pdf)
df_with_embeddings.write.format("delta").mode("overwrite").saveAsTable("default.gramseva_embeddings")

print("✅ gramseva_embeddings table saved to Delta Lake!")

# COMMAND ----------

# STEP 6: Verify embeddings table
result = spark.table("default.gramseva_embeddings")
print(f"✅ Total rows with embeddings: {result.count()}")
print("\n📊 Sample embeddings data:")
result.select("scheme_name", "category", "level", "embedding").show(5, truncate=50)

print("\n🎉 Embeddings generation complete! You can now run gramseva_03_rag.py")