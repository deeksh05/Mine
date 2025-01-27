import org.apache.spark.sql.{DataFrame, SparkSession, Column}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Tokenizer, HashingTF}
import org.apache.spark.ml.linalg.{SparseVector, DenseVector}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import scala.collection.immutable.Map

val kuduMaster: String = "tblpjc0v0400c-hdp.verizon.com,tblpjc0v0401c-hdp.verizon.com,tblpjc0v0402c-hdp.verizon.com"
val sparkConf:SparkConf = new SparkConf().setAppName("DuplicateAddress")
val spark: SparkSession = SparkSession.builder()
                .config(sparkConf)
                .config("spark.kudu.master", kuduMaster)
                .getOrCreate()

val inventoryKuduTable = "impala::mns_kuduaiops.fwa_combined_inventory_full"
val latLongKuduTable = "impala::mns_kuduaiops.fwa_customer_locations"
val abbrev_file =  "/edl/hdfs/jffv-mns/address/input/abbrev.csv"

val output_path = "/edl/hdfs/jffv-mns/address/output"
val temp_path = "/edl/hdfs/jffv-mns/address/temp"

val loacations_path = "/edl/hdfs/jffv-mns/address/input/fwa_customer_locations.csv"
val inventory_path = "/edl/hdfs/jffv-mns/address/input/fwa_combined_inventory_full.csv"

// val inventoryDf = spark.read.format("kudu")
//                     .option("kudu.master", kuduMaster)
//                     .option("kudu.table", inventoryKuduTable)
//                     .load();


// val latLongDf = spark.read.format("kudu")
//                     .option("kudu.master", kuduMaster)
//                     .option("kudu.table", latLongKuduTable)
//                     .load();



val inventoryDf = spark.read.format("csv").option("header", "true").load(inventory_path)
val latLongDf = spark.read.format("csv").option("header", "true").load(loacations_path)

val cleanTokensUDF = udf((tokens: Seq[String]) => tokens.map(_.trim))
val lowerCaseUDF = udf((tokens: Seq[String]) => tokens.map(_.toLowerCase))
val removeNullUDF = udf((tokens: Seq[String]) => tokens.map(_.trim).filter(_.nonEmpty))  

def cleanAddress(src_data: DataFrame): DataFrame = {
    val splitted_add = src_data.withColumn("address_tokens", split(col("address"), ","))    
    val cleanedData = splitted_add.withColumn("address_tokens", removeNullUDF(lowerCaseUDF(cleanTokensUDF(col("address_tokens")))))
    cleanedData
}

// Load and process abbreviation data
val abbrev_df = spark.read.option("header", "true").csv(abbrev_file)
.withColumn("key", concat_ws(",", col("Comman"), col("Standard")))
.withColumn("key", split(col("key"), ","))
.withColumn("key", lowerCaseUDF(cleanTokensUDF(col("key"))))
.withColumn("key", explode(col("key")))
.withColumn("value", lower(col("Primary")))
.select("key", "value")

// Broadcast the abbreviation map for efficient lookup
val abbrev_map = spark.sparkContext.broadcast(abbrev_df.rdd.map(row => (row.getString(0), row.getString(1))).collectAsMap().toMap)

def replaceTokensInSentence(sentence: String, replacements: Map[String, String]): String = {
  replacements.foldLeft(sentence) {
      case (updatedSentence, (key, value)) => updatedSentence.replaceAll(s"\\b$key\\b", value)
  }
}

// UDF to replace tokens using broadcasted abbreviation map
val replaceTokensUDF: UserDefinedFunction = udf((tokens: Seq[String]) => {
if (tokens != null) tokens.map(sentence => replaceTokensInSentence(sentence, abbrev_map.value)) else tokens
})

val replaceTokenUDF: UserDefinedFunction = udf((sentence: String) => {
  if (sentence != null) replaceTokensInSentence(sentence, abbrev_map.value) else sentence
})

// inventoryDf.select("site_id", "address", "postal_code" ).limit(10000).write.mode("overwrite").format("parquet").save(s"${temp_path}/inventoryDf")

// val inventorySrcDf_stg = cleanAddress(spark.read.format("parquest").load(s"${temp_path}/inventoryDf"))
val inventorySrcDf_stg = cleanAddress(inventoryDf.select("site_id", "address", "postal_code" ))
    .select(
        monotonically_increasing_id().alias("row_id"),
        col("site_id"),
        replaceTokenUDF(trim(col("address_tokens").getItem(0))).alias("address_line1"),
        replaceTokenUDF(trim(col("address_tokens").getItem(1))).alias("address_line2"),
        trim(col("address_tokens").getItem(2)).alias("city"),
        trim(col("address_tokens").getItem(3)).alias("state_or_province"),
        trim(col("address_tokens").getItem(4)).alias("parsed_postal_code"),
        trim(col("address_tokens").getItem(5)).alias("country"),
        col("postal_code"),
        col("address").alias("orig_address")
    ).withColumn("address", concat_ws(",", col("address_line1"), coalesce(col("address_line2"), lit(""))))

inventorySrcDf_stg.write.mode("overwrite").format("parquet").save(s"${temp_path}/inventorySrcDf")

// latLongDf.select("row_id", "address_line1", "address_line2", "city", "state_or_province", "country", "postal_code", "latitude", "longitude")
//   .dropDuplicates().limit(10000)
//   .write.mode("overwrite").format("parquet").save(s"${temp_path}/latLongDf")

// val latLongSrcDf_stg = spark.read.format("parquest").load(s"${temp_path}/latLongDf")

val latLongSrcDf_stg = latLongDf
.select(
    monotonically_increasing_id().alias("row_id"),
    col("address_hashcode"),
    replaceTokenUDF(trim(lower(col("address_line1")))).alias("address_line1"), 
    replaceTokenUDF(trim(lower(col("address_line2")))).alias("address_line2"),
    trim(lower(col("city"))).alias("city"),
    trim(lower(col("state_or_province"))).alias("state_or_province"),
    trim(lower(col("country"))).alias("country"),
    trim(lower(col("postal_code"))).alias("postal_code"),
    col("latitude"),
    col("longitude")
).withColumn("address", concat_ws(",", col("address_line1"), coalesce(col("address_line2"), lit(""))))

latLongSrcDf_stg.write.mode("overwrite").format("parquet").save(s"${temp_path}/latLongSrcDf")



val inventorySrcDf = cleanAddress(spark.read.format("parquet").load(s"${temp_path}/inventorySrcDf"))
val latLongSrcDf = cleanAddress(spark.read.format("parquet").load(s"${temp_path}/latLongSrcDf"))


// Feature extraction using HashingTF
val hashingTF = new HashingTF()
  .setInputCol("address_tokens")
  .setOutputCol("rawFeatures")
  .setNumFeatures(1000)

def cosineSimilarity(vec1: Array[Double], vec2: Array[Double]): Double = {
  val dotProduct = vec1.zip(vec2).map { case (x, y) => x * y }.sum
  val normA = math.sqrt(vec1.map(x => x * x).sum)
  val normB = math.sqrt(vec2.map(x => x * x).sum)
  if (normA > 0 && normB > 0) dotProduct / (normA * normB) else 0.0
}

val cosineSimilarityUDF = udf((vec1: SparseVector, vec2: SparseVector) => cosineSimilarity(vec1.toArray, vec2.toArray))

val featurizedData_inv = hashingTF.transform(inventorySrcDf)
val featurizedData_loc = hashingTF.transform(latLongSrcDf)

// Perform self-join to find similar addresses
val joinedData = featurizedData_inv.as("a")
  .join(featurizedData_loc.as("b"),
    col("a.postal_code") === col("b.postal_code"),
    "left"
  )


val windowSpec = Window.partitionBy(col("RowID_inv")).orderBy(col("SimilarityScore").desc)
// Calculate similarity and filter results
val similarityThreshold = 0.9
val similarityData_stg = joinedData.select(
  col("a.row_id").alias("RowID_inv"),
  col("b.row_id").alias("RowID_ll"),
  col("a.address").alias("Address_inv"),
  col("b.address").alias("Address_ll"),
  col("b.address_hashcode").alias("address_hashcode"),
  col("a.site_id").alias("site_id"),
  col("a.orig_address").alias("address"),
  col("a.postal_code").alias("postal_code"),
  col("b.latitude").alias("latitude"),
  col("b.longitude").alias("longitude"),
  when(col("b.row_id").isNotNull, 
    cosineSimilarityUDF(
      col("a.rawFeatures"),
      col("b.rawFeatures")
    )).otherwise(-1).alias("SimilarityScore")
).withColumn("rnk", row_number().over(windowSpec)).where(col("rnk")===1).drop("rnk")


similarityData_stg.write.mode("overwrite").format("parquet").save(s"${temp_path}/similarityData")
val similarityData = cleanAddress(spark.read.format("parquet").load(s"${temp_path}/similarityData"))

val exact_matched_df = similarityData.where(col("SimilarityScore") >= 0.9).select(
  // col("RowID_inv"),
  col("site_id"),
  col("address_hashcode"),
  col("address"),
  col("postal_code"),
  col("latitude"),
  col("longitude"),
  // col("Address_ll"),
  lit("exact").alias("match"),
  col("SimilarityScore")
)

val similar_matched_df = similarityData.where((col("SimilarityScore") >= 0 ) && (col("SimilarityScore") < 0.9)  ).select(
   // col("RowID_inv"),
  col("site_id"),
  col("address_hashcode"),
  col("address"),
  col("postal_code"),
  col("latitude"),
  col("longitude"),
  // col("Address_ll"),
  lit("similar").alias("match"),
  col("SimilarityScore")
)

val no_matched_df = similarityData.where(col("SimilarityScore") < 0).select(
  // col("RowID_inv"),
  col("site_id"),
  col("address_hashcode"),
  col("address"),
  col("postal_code"),
  col("latitude"),
  col("longitude"),
  // col("Address_ll"),
  lit("no match").alias("match"),
  col("SimilarityScore")
)

val df_to_save = exact_matched_df.unionByName(similar_matched_df).unionByName(no_matched_df)

df_to_save.coalesce(1).write.mode("overwrite").format("csv").option("header", "true").save(output_path)

