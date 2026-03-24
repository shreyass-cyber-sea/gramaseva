# Databricks notebook source
# TITLE: GramSeva - Step 1: Data Setup
# Download dataset from Kaggle and create Delta tables

# STEP 1: Install kagglehub and download dataset
# MAGIC %pip install kagglehub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import kagglehub
import os

print("📥 Downloading Indian Government Schemes dataset from Kaggle...")

# Download latest version from Kaggle
try:
    path = kagglehub.dataset_download("jainamgada45/indian-government-schemes")
    print("✅ Path to dataset files:", path)

    # List files in the downloaded directory
    dataset_files = os.listdir(path)
    print("📁 Available files:", dataset_files)

    # Find the main CSV file (usually updated_data.csv)
    csv_file = None
    for file in dataset_files:
        if file.endswith('.csv') and 'updated' in file.lower():
            csv_file = os.path.join(path, file)
            break

    if not csv_file:
        # Fallback to any CSV file
        csv_files = [f for f in dataset_files if f.endswith('.csv')]
        if csv_files:
            csv_file = os.path.join(path, csv_files[0])
            print(f"📄 Using CSV file: {csv_files[0]}")

    if not csv_file:
        raise FileNotFoundError("No CSV file found in the downloaded dataset")

    print(f"✅ Dataset file ready: {csv_file}")

except Exception as e:
    print("❌ Error downloading from Kaggle:", str(e))
    print("Please ensure:")
    print("1. Internet connection is available")
    print("2. Kaggle dataset URL is correct")
    print("3. No firewall blocking Kaggle access")
    raise e

# COMMAND ----------

# STEP 2: Load CSV file into Spark DataFrame
print("📊 Loading dataset into Spark DataFrame...")

try:
    # Read the CSV file directly from the downloaded path
    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(f"file://{csv_file}")
    print(f"✅ Loaded {df_raw.count()} rows from Kaggle dataset")
    # Save as Delta table
    df_raw.write.format("delta").mode("overwrite").saveAsTable("default.updated_data")
    print("✅ Created default.updated_data table")

    # Show basic info about the dataset
    print(f"📊 Dataset schema:")
    df_raw.printSchema()
    print(f"\n📋 Sample data:")
    df_raw.show(3, truncate=True)

except Exception as e:
    print("❌ Error loading dataset:", str(e))
    raise e

# COMMAND ----------

# STEP 3: Create a clean GramSeva schemes table
spark.sql("""
  CREATE TABLE IF NOT EXISTS default.gramseva_schemes
  AS SELECT * FROM default.updated_data
""")

print("✅ gramseva_schemes table ready!")

# COMMAND ----------

# STEP 4: Verify table creation and show sample data
df = spark.table("default.gramseva_schemes")
print(f"✅ Total schemes in table: {df.count()}")
print("\n📊 Table schema:")
df.printSchema()
print("\n📋 Sample data:")
df.show(3, truncate=False)

print("\n🎉 Data setup complete! You can now run gramseva_02_embeddings.py")