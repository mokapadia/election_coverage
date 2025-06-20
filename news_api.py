

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import matplotlib.patches as mpatches
from scipy.stats import f_oneway, shapiro, levene



# class that represents News API, use to make calls and query data in Mongo
class NewsApi:
    
    # create a client
    def __init__(self):

        # connect to mongo
        client = MongoClient()

        # establish your database
        self.db = client['hw6']

        # establish the collection
        self.collection = self.db['news']
        

 
    def articles(self):
        """ 
        plot the distribution of articles across the different sources
        """
        
        # Define aggregation pipeline
        pipeline = [
            {"$group": {"_id": '$source.name', "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},  
            {"$limit": 10}  ]
            
        # Perform aggregation
        res = self.collection.aggregate(pipeline)
        
        # Convert to pandas DataFrame
        mongo_df = pd.DataFrame(res)
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x='_id', y='count', data=mongo_df, color='#87CEEB')
        plt.title('Distribution of Articles Across Different Sources')
        plt.xlabel('Source')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return list(mongo_df["_id"])
    

    def time_count(self):
        """ plot the number of articles over time """
        res = list(self.collection.aggregate([{"$project": {"publishedAt":
                                                            {"$substr": ["$publishedAt", 0, 10]}}}]))
        res = [datetime.strptime(i["publishedAt"], "%Y-%m-%d") for i in res]
        dct = Counter(res)
        plt.plot_date(dct.keys(), dct.values())
        plt.title("Temporal trends in Articles over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation = 45)
        plt.show() 
        
        

    def get_top_words(self):
        """find the top words in all the articples and plot the word cloud"""
        nltk.download('stopwords')

        
        pipeline = [ # unwind content array and split into words
                    {"$unwind": "$content"}, 
                    {"$project": {"words": {"$split": ["$content", " "]}}},
                    {"$unwind": "$words"},
                    
                    # count the instances of each word
                    {"$group": {"_id": "$words", "count": {"$sum": 1}}}, 
                    
                    # filter out list of stop words to only include relevant, meaningful words
                    {"$match": {"_id": {"$nin": stopwords.words('english') +["de", "et"]}}},
                    
                     # sort in descending order
                    {"$sort": {"count": -1}},
                    
                     # limit to top 10 words
                    {"$limit": 10}]
        
        # Perform aggregation
        res = self.collection.aggregate(pipeline)

        # Convert to pandas DataFrame
        mongo_df = pd.DataFrame(res)
        
        # cinvert to dictionary 
        word_dict = dict(zip(mongo_df['_id'], mongo_df['count']))
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
    
        # plot cloud 
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Top Words')
        plt.show()
     

        
    
    # gets the character count of each article
    def character_count(self):
        """
        Create histogram of character counts across articles
        - 77 articles that were skewing data
        
        """
        res = list(self.collection.find({"character_count": {"$exists": "true", "$lt": 40000}}, 
                                        {"_id":0, "character_count":1}))
        counts = [i['character_count'] for i in res if i["character_count"] is not None]
        plt.hist(counts, bins = 25)
        plt.title("Character Count Distribution Across Articles (under 40000)")
        plt.xlabel("Character Count")
        plt.ylabel("Number of Articles")
        
        plt.show()
    
    
    # give insight to how many articles are of trump, biden etc
    def topic_analyzer(self):
        """
            Pie Chart Analyzing Categories """
        
        pipeline = [{"$group": {"_id": '$category', "count": {"$sum": 1}}}]

        res = self.collection.aggregate(pipeline)


        # Convert to pandas DataFrame
        mongo_df = pd.DataFrame(res)
        
        
        # make a pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(mongo_df["count"], labels=mongo_df["_id"], autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Documents in Each Category')
        plt.axis('equal')  
        plt.show()
        
        
    def sentiment(self, source_lst, avg = None, keyword = None):
        """
            return sentiment scores based on limited sources
            w/ averages 
            helper function 
        """
       
        
    
        if avg:
            pipeline = [{"$match": {'source.name': {"$in": source_lst}}}, {"$group": {"_id": '$source.name', "avg": {"$avg": "$sentiment"}}}]
            res = list(self.collection.aggregate(pipeline))

            sentiment = {}
            sentiment["source"] = [i["_id"] for i in res]
            sentiment["avg"] = [i["avg"] for i in res]



        else:                           
            res = list(self.collection.find({'source.name': {"$in": source_lst}}, {"_id":0, 'source.name':1, 'sentiment':1}))
            sentiment = {}
            sentiment["source"] = [i["source"]["name"] for i in res]
            sentiment["score"] = [i["sentiment"] for i in res]




            if keyword:
                res = list(self.collection.find({"source.name": {"$in": source_lst}, "category": keyword}, {"_id": 0, "source.name":1, "sentiment":1, "category":1}))
                sentiment = {}
                sentiment["source"] = [i["source"]["name"] for i in res]
                sentiment["score"] = [i["sentiment"] for i in res]
                sentiment["category"] = [i["category"] for i in res]
                


        return pd.DataFrame(sentiment)
            
            
            
        
    def overall_hist(self):
        """"""
        res = list(self.collection.find({ }, {"_id":0, 'sentiment':1}))
        counts = [i["sentiment"] for i in res]
        plt.hist(counts)
        plt.title("Distribution of Sentiment Scores Across Sources")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Number of Articles")
        plt.show()


    def compare_hist(self, right, left):
        """create overlayed"""
        
        right_res = self.sentiment(right)
        left_res = self.sentiment(left)
        
        plt.hist(right_res["score"], alpha =0.5, label = "Right-Leaning", color = "red")
        plt.hist(left_res["score"], alpha =0.5, label = "Left-Leaning", color = "blue")
        plt.legend()
        plt.title("Distribution of Sentiment Scores in Right & Left-Leaning Sources")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.show()
        
        
    def centrist_hist(self, centrist, source = None):
        """
        histogram across centrist new sources
        
        """
        
        centrist_df = self.sentiment(centrist)
        
        if source:
            
            group = self.sentiment(centrist, avg = True)
            plt.bar(group["source"], group["avg"], color = ["purple"] * len(group))
            plt.xlabel("Sources")
            plt.ylabel("Average Sentiment Score")
            plt.ylim((-.4, .4))
            plt.axhline(y= centrist_df["score"].mean(axis =0), color='gray', linestyle='--', 
                        linewidth=2, label='Average Centrist Sentiment Score')
            plt.legend()
            plt.xticks(rotation = 90)
            plt.title("Average Sentiment Scores across Centrist New Sources")
            
                  
            
        else:  
            
            plt.hist(centrist_df["score"], color = "purple")
            plt.title("Distribution of Sentiment Scores in Centrist Sources")
            plt.xlabel("Sentiment Score")
            plt.ylabel("Frequency")
        
        
        plt.show()
        
        
    def left_vs_right(self, right, left, center, source = None):
        """
        comparing bar charts either average across political views or sources
        
        """
        
        
        if source:
            group = pd.concat([self.sentiment(right, avg = True), self.sentiment(left, avg = True)])
        
            color = ["Red" if i in right else "Blue" for i in group["source"]] 
            
            plt.bar(group["source"], group["avg"], color = color)
            plt.xlabel("Sources")
            plt.ylabel("Average Sentiment Score")
            plt.ylim((-.4, .4))
            
            red_patch = mpatches.Patch(color='red', label='right-wing')
            blue_patch = mpatches.Patch(color='blue', label='left-wing')

            plt.legend(handles=[red_patch, blue_patch])
            
            plt.xticks(rotation = 90)
            plt.title("Average Sentiment Scores across Top Biased Sources")
            
            
            
        else:
            
            right_df = self.sentiment(right)
            right_df.insert(2, "view",["Right"] * len(right_df) , True)
    
            
            left_df = self.sentiment(left)
            left_df.insert(2, "view",["Left"] * len(left_df) , True)
            
            center_df = self.sentiment(center)
            center_df.insert(2, "view",["center"] * len(center_df) , True)
        
            total_df = pd.concat([right_df, center_df, left_df])
            
            group = total_df.groupby(["view"]).mean('score')
            
            plt.bar(group.index, group["score"], color = ["Blue", "Red", "Purple",])
            plt.xlabel("Bias of Sources")
            plt.ylabel("Average Sentiment Score")
            plt.title("Average Sentiment Scores across Political Affiliations")
           
            
            
        
        plt.show()
        
    
    
    
    def bias_by_topic(self, right, left, keyword = None):
        """
        scatter plot by keyword right v left 
        
        """
        if keyword:
            total_df = self.sentiment(right + left, keyword = keyword)
            plt.title(f"Seniment Scores of Articles in {keyword} Category - Right vs. Left")
        
        else:
            total_df = self.sentiment(right + left)
            plt.title(f"Seniment Scores of Articles in ALL Categories - Right vs. Left")
        
        color = ["Red" if i in right else "Blue" for i in total_df["source"]] 
        plt.scatter(range(len(total_df)), total_df["score"], color = color)
        
        
        red_patch = mpatches.Patch(color='red', label='right-wing')
        blue_patch = mpatches.Patch(color='blue', label='left-wing')

        plt.legend(handles=[red_patch, blue_patch])
        plt.ylabel("Sentiment Score")
        plt.xlabel("Article Index")
    
        plt.show()
        
        
    # compares two sentiment scores of two sources 
    def compare_two_sources(self, source1, source2, category= None):
            
        if category:
            total_df = self.sentiment([source1, source2], keyword = category)
            plt.title(f"Seniment Scores of Articles in {category} Category - {source1} vs. {source2}")
        else:
            total_df = self.sentiment([source1,source2])
            plt.title(f"Seniment Scores of Articles in ALL Categories - {source1} vs. {source2}")
                
            
        colors = ['red' if source == source1 else 'blue' for source in total_df['source']]
    
        plt.scatter(range(len(total_df)), total_df["score"], color=colors)
        
        red_patch = mpatches.Patch(color='red', label=f'{source1}')
        blue_patch = mpatches.Patch(color='blue', label= f'{source2}')

        plt.legend(handles=[red_patch, blue_patch])
        plt.ylabel("Sentiment Score")
        plt.xlabel("Article Index")
    
        plt.show()
        
    
   
        
    def anova(self, right, left, center):
        """ conduct ANOVA between all three politically affiliated 
            sources sentiment scores averages
            
            plot boxplot
        """
        
        
        # aggregate average scores based on different affiliations
        centrist_scores = self.sentiment(center, avg = True)
        left_scores = self.sentiment(left, avg = True)
        right_scores = self.sentiment(right, avg = True)


        # CHECK ASSUMPTIONS OF ANOVA
        # Check for normal distribution within each group using Shapiro-Wilk test
        # p values should be greater than 0.05 to reject null hypothesis
        # extract p value
        p1 = shapiro(centrist_scores["avg"])[1]
        p2 = shapiro(left_scores["avg"])[1]
        p3 = shapiro(right_scores["avg"])[1]

        # Check for homogeneity of variances using levene test
        p_var = levene(centrist_scores["avg"], left_scores["avg"], right_scores["avg"])[1]

        # Perform ANOVA if assumptions are met
        if (p1 > 0.05) & (p2 > 0.05) & (p3 > 0.05) & (p_var > 0.05):
            f_statistic, p_value = f_oneway(centrist_scores["avg"], left_scores["avg"], right_scores["avg"])
            print("ANOVA results:")
            print("F-statistic:", f_statistic)
            print("p-value:", p_value)
            if p_value < 0.05:
                print("Reject the null hypothesis: There are significant differences in \
                      average sentiment scores between the news sources across political affiliations.")
            else:
                print("Fail to reject the null hypothesis: There is no significant difference in \
                      average sentiment scores between the news sources across political affiliations.")


            # Create boxplot
            total_df = pd.concat([left_scores, centrist_scores, right_scores])
            affiliation = (["left-wing"] * len(left_scores)) + (["centrist"] * len(centrist_scores)) + (["right-wing"] * len(right_scores)) 

            total_df.insert(2, "affiliation", affiliation, True)
            my_pal = {"right-wing": "r", "left-wing": "b", "centrist":"m"}

            sns.boxplot(x=total_df["affiliation"], y = total_df["avg"], palette = my_pal)

            plt.title('Boxplot of Average Sentiment Scores Each Political Affiliation')
            plt.xlabel('Political Affiliation')
            plt.ylabel('Sentiment Score')
            plt.show()

