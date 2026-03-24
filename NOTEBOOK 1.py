# Databricks notebook source
# STEP 1: Load CSV data from DBFS into DataFrame first
# Make sure you've uploaded updated_data.csv to /FileStore/gramseva/ in Databricks

# Check if file exists
try:
    dbutils.fs.ls("/FileStore/gramseva/updated_data.csv")
    print("✅ CSV file found in DBFS")
except Exception as e:
    print("❌ CSV file not found. Please upload updated_data.csv to /FileStore/gramseva/ in Databricks")
    print("Go to Data -> Create Table -> Upload File -> Browse -> Select updated_data.csv")
    raise FileNotFoundError("Please upload the dataset first")

# Load CSV file from DBFS
df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/gramseva/updated_data.csv")
print(f"✅ Loaded {df_raw.count()} rows from CSV")

# Save as Delta table
df_raw.write.format("delta").mode("overwrite").saveAsTable("default.updated_data")
print("✅ Created default.updated_data table")

# Create a clean GramSeva table from the uploaded data
spark.sql("""
  CREATE TABLE IF NOT EXISTS default.gramseva_schemes
  AS SELECT * FROM default.updated_data
""")

print("✅ gramseva_schemes table ready!")

# COMMAND ----------

# Verify your table loaded correctly
df = spark.table("default.gramseva_schemes")
print(f"Total schemes: {df.count()}")
df.printSchema()
df.show(3)