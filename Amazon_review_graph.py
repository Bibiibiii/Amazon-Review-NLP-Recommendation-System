
# coding: utf-8

# In[2]:


#spark sql imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport().appName('ReadWriteData').getOrCreate()
sc = spark.sparkContext


# In[4]:


df=spark.read.json("/user/mxl6731/data/5core_sample.json.gz")
df.printSchema()


# In[5]:


df.show(5, vertical=True)


# In[6]:


from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from graphframes import *


# In[7]:


spark = SparkSession.builder     .appName("GraphFramesAPI")    .getOrCreate()


#  asin           | 1384719342           
#  helpful        | [13, 14]             
#  overall        | 5.0                  
#  reviewText     | The product does ... 
#  reviewTime     | 03 16, 2013          
#  reviewerID     | A14VAT5EAX3D9S       
#  reviewerName   | Jake                 
#  summary        | Jake                 
#  unixReviewTime | 1363392000 

# ## Building the graph

# In[8]:


df = (df
        .withColumnRenamed('asin', 'productID')
        .withColumnRenamed('overall', 'rating'))


# In[9]:


products = (df
            .selectExpr('productID as id')
            .withColumn('reviewerName', F.lit(None))
            .distinct())
reviewers = df.selectExpr('reviewerID as id', 'reviewerName').distinct()#.withColumn('id', df['reviewerID'])
# .withColumn('id', F.monotonically_increasing_id())
products.show()


# In[10]:


vertices = products.union(reviewers)


# In[11]:


vertices.count()


# In[12]:


# unioned = (products
#            .unionByName(reviewers)
#            .withColumn('overall', F.when(F.col('attribute').rlike('\d\.\d'), F.col('attribute')).otherwise(None))
#            .withColumn('reviewerName', F.when(~F.col('attribute').rlike('\d\.\d'), F.col('attribute')).otherwise(None))
#            .drop('attribute')
#           )


# In[13]:


# data cleanning 
vertices = (vertices
           .withColumn('reviewerName', F.trim(F.regexp_extract('reviewerName', '([a-zA-Z0-9_\s\.\']+)', 1)))
            # .withColumn('new_name', F.regexp_extract('reviewerName', '(.+)\s["|$]', 1))
#             .filter(F.col('reviewerName').isNotNull())
            )


# In[14]:


vertices.filter(F.col('reviewerName').isNotNull()).show()


# In[15]:


# products = df.select('asin', 'overall').withColumn('id', df['asin'])
# reviewers = df.select('reviewerID', 'reviewerName').withColumn('id', df['reviewerID'])
# vertices = products.unionByName(reviewers, allowMissingColumns=True)
# nodes = df.select('asin', 'overall').withColumn('id', df['asin'])
review_1 = (df
          .select('reviewerID', 'productID', 'rating', 'helpful', 'reviewTime', 'reviewText')
          .withColumn('src', df['reviewerID'])
          .withColumn('dst', df['productID']))
# review.show(5)


# In[16]:


review_2 = (df
          .select('reviewerID', 'productID', 'rating', 'helpful', 'reviewTime', 'reviewText')
          .withColumn('src', df['productID'])
          .withColumn('dst', df['reviewerID']))
# review.show(5)


# we need to duplicate the dataset here because we want bi-directional graph

# In[17]:


reviews = review_1.union(review_2)


# In[18]:


from graphframes import *


# In[19]:


graph = GraphFrame(vertices, reviews)


# In[20]:


graph.vertices.show(5)


# In[21]:


graph.edges.show()


# In[22]:


graph.degrees.show()


# In[23]:


graph.inDegrees.show()


# In[24]:


graph.outDegrees.show()


# In[25]:


# ## Display all connected components
# sc = spark.sparkContext
# sc.setCheckpointDir('graphframes_cps')
# graph.connectedComponents().show()


# ## Filtering 

# In[26]:


## Filter and sort vertices with degree >=2
x = graph.degrees.filter("degree >= 2").sort("degree", ascending=False)
x.count()


# In[27]:


x = graph.outDegrees.filter("outDegree >= 2").sort("outDegree", ascending=False)
x.count()


# In[28]:


graph.find("(a)-[e]->(b); (c)-[e2]->(b)").show(truncate = False)


# In[29]:


graph.find("(a)-[e]->(b); (b)-[e2]->(c)").show(truncate = False, vertical=True)


# In[30]:


#To query all the mutual friends between 2 and 3 we can filter the DataFrame

mutualFriends =  graph.find("(a)-[]->(b); (b)-[]->(c); (c)-[]->(b); (b)-[]->(a)").dropDuplicates()
mutualFriends.filter('a.id == 2 and c.id == 3').show(truncate = False)
#mutualFriends.show(truncate = False)


# In[31]:


graph.triangleCount().show()


# In[32]:


graph.vertices.filter("id='B000MWWT6E'").show()


# In[33]:


graph.edges.filter("src='A2Z3FDWL7DQD76'").show()


# In[34]:


graph.edges.filter("dst='B000MWWT6E'").show()


# In[35]:


graph.triangleCount().filter(F.col('count')>0).show()


# In[36]:


graph.triangleCount().show()


# In[37]:


graph.edges.groupBy("productID").count().orderBy("count", ascending = False).show()


# In[38]:


# Products has the most reviews
(graph.edges
    .groupBy("src")
    .count()
    .orderBy("count", ascending = False)
    .withColumnRenamed('src', 'productID')
    .show())


# In[39]:


# pr = graph.pageRank(resetProbability=0.15, maxIter=10)## look at the pagerank score for every vertex
# pr.vertices.show()

# ## look at the weight of every edge
# pr.edges.show()


# In[40]:


graph.bfs("id = 'B003VWJ2K8'" , "id = 'B003VWKPHC'", maxPathLength = 10).show(truncate = False)


# In[41]:


graph.bfs("id = 'AE9C0UNXBV8CB'" , "id = 'A2Y7BSQG9V3LNG'", maxPathLength = 10).show(truncate = False)


# In[43]:


# shortest path
# paths = graph.shortestPaths(landmarks=("A2IBPI20UZIR0U", "A195EZSQDW3E21"))
# paths.show()


# In[ ]:


# result = graph.labelPropagation(maxIter=5)
# result.show()


# a   | [A2IBPI20UZIR0U, cassandra tu]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
#  e   | [A2IBPI20UZIR0U, 1384719342, 5.0, [0, 0], 02 28, 2014, Not much to write about here, but it does exactly what it's supposed to. filters out the pop sounds. now my recordings are much more crisp. it is one of the lowest prices pop filters on amazon so might as well buy it, they honestly work the same despite their pricing,, A2IBPI20UZIR0U, 1384719342]                                                                                                                                                                                                                                                                                       
#  b   | [1384719342,]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#  e2  | [A195EZSQDW3E21, 1384719342, 5.0, [1, 1], 08 28, 2013, The primary job of this device is to block the breath that would otherwise produce a popping sound, while allowing your voice to pass through with no noticeable reduction of volume or high frequencies. The double cloth filter blocks the pops and lets the voice through with no coloration. The metal clamp mount attaches to the mike stand secure enough to keep it attached. The goose neck needs a little coaxing to stay where you put it., 1384719342, A195EZSQDW3E21]                                                                                                               
#  c   | [A195EZSQDW3E21, Rick Bennette]   

# In[ ]:


# !SPARK_HOME/bin/spark-shell --jars neo4j-connector-apache-spark_2.12-4.0.1_for_spark_3.jar


# In[ ]:


# spark.read.format("org.neo4j.spark.DataSource") \
#   .option("url", "bolt://localhost:7687") \
#   .option("labels", "Person:Customer:Confirmed") \
#   .load()


# Next: 
# 1. deploy on local for PageRank and ConnectedComponents.
# 2. Tried to set up on databrike for neo4j connection. It's a good way for data visualization.
