# Databricks notebook source
# TITLE: GramSeva - Step 1: Data Setup
# Load CSV data from DBFS and create Delta tables

# STEP 1: Verify CSV file exists in DBFS
try:
    dbutils.fs.ls("/FileStore/gramseva/updated_data.csv")
    print("✅ CSV file found in DBFS")
except Exception as e:
    print("❌ CSV file not found. Please upload updated_data.csv to /FileStore/gramseva/ in Databricks")
    print("Steps to upload:")
    print("1. Go to Data -> Create Table -> Upload File")
    print("2. Browse and select updated_data.csv")
    print("3. Choose destination: /FileStore/gramseva/")
    raise FileNotFoundError("Please upload the dataset first")

# COMMAND ----------

# STEP 2: Load CSV file from DBFS
print("Loading CSV data from DBFS...")
df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/gramseva/updated_data.csv")
print(f"✅ Loaded {df_raw.count()} rows from CSV")

# Save as Delta table
df_raw.write.format("delta").mode("overwrite").saveAsTable("default.updated_data")
print("✅ Created default.updated_data table")

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