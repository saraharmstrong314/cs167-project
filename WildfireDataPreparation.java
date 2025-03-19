package edu.ucr.cs.cs167.sdera006;

import edu.ucr.cs.bdlab.beast.JavaSpatialSparkContext;
import org.apache.spark.beast.CRSServer;
import org.apache.spark.beast.SparkSQLRegistration;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

public class App {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: WildfireDataPreparation <wildfireCSV> <countyShapefileZip> <outputParquet>");
            System.exit(1);
        }

        String wildfireCSV = args[0];
        String countyZip = args[1];
        String outputParquet = args[2];

        SparkConf conf = new SparkConf()
                .setAppName("Wildfire Data Preparation (Spark + Beast)")
                .setMaster("local[*]");


        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();

        // 3. Create the Beast JavaSpatialSparkContext from sparkSession
        JavaSpatialSparkContext sparkContext = new JavaSpatialSparkContext(spark.sparkContext());

        // 4. (Optional) Start the CRS server if you need coordinate transformations
        CRSServer.startServer(sparkContext);

        // 5. Register Beast's geometry UDT and UDF
        //    This is where ST_Point, ST_Contains, etc. become known to Spark
        SparkSQLRegistration.registerUDT();
        SparkSQLRegistration.registerUDF(spark);


        // 3. Load the Wildfire CSV
        //    Spark can handle .bz2, .gz, .zip, etc. automatically if you just give the filename.
        Dataset<Row> wildfiresRaw = spark.read()
                .option("header", "true")        // first line is header
                .option("inferSchema", "false")  // we will manually cast
                .option("delimiter", "\t")
                .csv(wildfireCSV);

        // System.out.println("Finished 3.");

        // 4. Keep only the needed columns
        Dataset<Row> wildfires = wildfiresRaw.select(
                col("x"),
                col("y"),
                col("acq_date"),
                col("frp"),
                col("acq_time"),
                col("ELEV_mean"),
                col("SLP_mean"),
                col("EVT_mean"),
                col("EVH_mean"),
                col("CH_mean"),
                col("TEMP_ave"),
                col("TEMP_min"),
                col("TEMP_max")
        );

        // 5. Fix the frp column (split by comma, cast first to double)
        //    e.g. "12.3,45.7" -> "12.3" -> double(12.3)
        wildfires = wildfires.withColumn(
                "frp",
                expr("double(split(frp, ',')[0])")
        );

        // 6. Cast numeric columns to double
        String[] numericCols = {
                "x", "y", "frp",
                "ELEV_mean", "SLP_mean", "EVT_mean", "EVH_mean",
                "CH_mean", "TEMP_ave", "TEMP_min", "TEMP_max"
        };
        for (String c : numericCols) {
            wildfires = wildfires.withColumn(c, col(c).cast("double"));
        }

        // 7. Create a geometry column using Beast’s ST_CreatePoint(x, y)
        wildfires = wildfires.withColumn(
                "geometry",
                expr("ST_CreatePoint(x, y)")
        );

        // System.out.println("Finished 7.");

        // 8. Load the county shapefile (zipped).
        //    Beast can read a zipped shapefile directly using format("shapefile").
        Dataset<Row> counties = spark.read()
                .format("shapefile")
                .load(countyZip)
                // keep geometry (the_geom) and GEOID
                .select(
                        col("geometry").alias("county_geom"),
                        col("GEOID")
                );

        // System.out.println("Finished 8.");

        // 9. Create temporary views to do a spatial join in SQL
        wildfires.createOrReplaceTempView("wildfires");
        counties.createOrReplaceTempView("counties");

        // 10. Spatial join using ST_Contains(county_geom, geometry)
        //     to find which county polygon covers each wildfire point
        String joinSQL = ""
                + "SELECT "
                + " w.x, w.y, w.acq_date, w.frp, w.acq_time, "
                + " w.ELEV_mean, w.SLP_mean, w.EVT_mean, w.EVH_mean, "
                + " w.CH_mean, w.TEMP_ave, w.TEMP_min, w.TEMP_max, "
                + " c.GEOID AS County, w.geometry "
                + "FROM wildfires w JOIN counties c "
                + "ON ST_Contains(c.county_geom, w.geometry)";

        Dataset<Row> joined = spark.sql(joinSQL);

        // System.out.println("Finished 10.");

        // 11. Drop geometry column from final output
        Dataset<Row> finalDF = joined.drop("geometry");

        // System.out.println("Finished 11.");
        // 12. Write as Parquet
        finalDF.write().mode("overwrite").parquet(outputParquet);

        // System.out.println("Finished 12.");

        System.out.println("Done! Created Parquet at: " + outputParquet);
        spark.stop();
    }
}
