#install.packages("installr")
library(installr)
#updateR()
# Make sure to update your R to 3.6.3
# ###################################################################
# Libraries
# ###################################################################
library(pacman)
pacman::p_load(tibble, rjson, xml2, tidytext, tokenizers, tidyr, dplyr, widyr, chron, openxlsx, reticulate, lubridate, stringr)
library(rjson)
library(xml2)
#install.packages("tidytext", repos='http://cran.us.r-project.org', dep = T, lib = "/home/app/R/x86_64-pc-linux-gnu-library/3.4")
#install.packages("tidytext", repos='http://cran.us.r-project.org', dep = T, lib = "C:/Users/maris/Documents/R/win-library/3.6")
library(tidytext)
library(tokenizers)
library(tidyr)
library(dplyr)
library(widyr)
library(chron) 
library(openxlsx) 
library(reticulate)
library(lubridate)
library(stringr)

# Shoudl topics be printed at the end
print_topics <- FALSE

# Minimal lambda value
v_min_lambda_daily <- 10

# Loading functions
# Setting main directory
#working_dir <- "/home/app/clustering/News_Clustering/" #for remote
working_dir <- "~/Sentinel/" #for local

# Sourcing R code
source(paste0(working_dir, "news-selectors/scripts/dunning_functions.R"), encoding = "UTF8")
source(paste0(working_dir, "news-selectors/scripts/text_cleaning_functions.R"), encoding = "UTF8")
source(paste0(working_dir, "news-selectors/scripts/topic_selection_functions.R"), encoding = "UTF8")

# ###################################################################
# Setting up Python integration
# ###################################################################
# Reticulate helps integrate
library(reticulate)
# Call the specific Python version with use_python or activate a conda env with use_condaenv
#path_to_python <- "C:/Users/maris/Anaconda3/envs/SBERT-WK/python"
#use_python(path_to_python)
# Using the same conda env as I do for Python
use_condaenv(condaenv = "SBERT-WK", conda = "/Users/maris/Anaconda3/envs/SBERT-WK", required = TRUE)

#scipy <- import("scipy")
source_python(paste0(working_dir, "news-selectors/scripts/python_functions.py"))

# Stop Words
source(paste0(working_dir, "news-selectors/scripts/PL_stop_words.R"), encoding = "UTF8")

#### LOAD DATA - this is what the input data has to look like
load(paste0(working_dir, 'news-selectors/data/daily_articles/archiv/articles_2019-07-08.RData'))
v_date = "2019-07-08"
####

# ###################################################################
# Load grammar dictionary - check out what this is 
# ###################################################################
load(paste0(working_dir, "news-selectors/data/grammar_data.RData"))

gc(reset = T)
# ###################################################################
# Modification of dataset
# ###################################################################
# TO DO:
# Correct sentence splitting (RMF24 - it's probably hard space)

# Force the "text" to be in UTF-8 to avoid errors in set_lambda_order in python_functions.py
Encoding(DF$text) <- "UTF-8"

# Selecting sentences that are to be included in analysis
articles_sentences <- DF %>%
    # Split into paragraphs
    unnest_tokens(text, text, token = "regex", pattern = " \\|\\| ", to_lower = F) %>%
    # Set paragraphs ids
    group_by(id) %>%
    mutate(paragraph_id = paste0(id, "_", seq(1, n()))) %>%
    ungroup() %>%
    # Split into sentences
    unnest_tokens(text, text, token = "sentences", to_lower = F) %>%
    # Delete too short sentences
    filter(nchar(text) > 80) %>%
    # Set sentences ids
    group_by(id) %>%
    mutate(sentence_id = paste0(id, "_", seq(1, n()))) %>%
    ungroup() %>%
    # Delete sentences with too many capital letters
    mutate(characters = nchar(text),
           capital_letters = stringr::str_count(text,"[A-Z]")) %>%
    filter((capital_letters / characters) < 0.35)

# Un-nest sentences
articles_unnested <- articles_sentences %>%
    unnest_tokens(word, text, to_lower = F) %>%
    group_by(sentence_id) %>%
    mutate(position = seq(1, n())) %>%
    ungroup()# %>%
    strip_numeric(word) %>% #(., word) %>% # commenting this out - removes digits from character string
    filter(!word %in% stop_words_pl) %>%
    left_join(grammar_data, by = c("word" = "tekst")) %>%
    mutate(word = ifelse(is.na(slowo), word, slowo)) %>%
    dplyr::select(-slowo) %>%
    filter(!word %in% stop_words_pl) 
    
articles_unnested

### IMPORTANT ERROR ###
#Warning message:
#In mask$eval_all_filter(dots, env_filter) : NAs introduced by coercion

# Select minimum number of tokens occurence basing on distribution of data
data_grouped <-  articles_unnested %>%
    group_by(word) %>%
    summarise(counts = n())
v_min_counts <- quantile(data_grouped$counts, probs = 0.95) # or 0.91

gc(reset = T)

# ###################################################################
# General statistics needed for Dunning statistic
# ###################################################################
# Data Set of dates used in general stats
used_dates <- read.csv2(paste0(working_dir, "news-selectors/data/Used_dates_in_stats.csv")) %>%
    dplyr::select(date)
# Statistics
load(paste0(working_dir, "news-selectors/data/General_stats_updated.RData"))

#####################################################################
# Cluster and summarise with embeddings in Python
# ###################################################################
# Prepare inputs for Python
inputs <- prepare_python_inputs(articles_unnested,
                                articles_sentences,
                                general_stats,
                                id, paragraph_id, sentence_id, word, 
                                v_min_counts, 
                                v_min_lambda_daily)

sections_and_articles <- inputs[[1]]
filtered_lambda_statistics <- inputs[[2]]
log_lambda_statistics_df <- inputs[[3]]
lemmatized_sentences <- inputs[[4]]
lemmatized_articles <- inputs[[5]]
sentences_text <- inputs[[6]]


# Sending data to Python for clustering and summarisation with the
# use of dimensions reduction (LSA)
# The algorithm is to select approximately 5% of the sentences that include 
# topic tokens (words), but no less than 3 and no more than 10.

############### DEBUGGERY ##########################

source_python(paste0(working_dir, "news-selectors/scripts/python_functions.py"))

#topics <- 
cluster_and_summarise(sections_and_articles, filtered_lambda_statistics,
                               # Clustering
                               min_association=0.25, do_silhouette=TRUE, 
                               singularity_penalty=-0.1,
                               # Summarization
                               lemmatized_sentences=lemmatized_sentences, 
                               lemmatized_articles=lemmatized_articles,
                               sentences_text=sentences_text,
                               log_lambda_statistics_df=log_lambda_statistics_df,
                               min_key_freq=0.25, max_sentence_simil=0.5,
                               section_id="section_id", word_col="word",
                               use_sparse=TRUE)
### IMPORTANT ERROR ###
#Error in py_call_impl(callable, dots$args, dots$keywords) : 
#KeyError: 'ALUFIRE' 

############### DEBUGGERY ##########################

list_topics <- topics[[1]]
words_similarity_matrix <- topics[[2]]
selected_tokens <- topics[[3]]
silhouette_history <- topics[[4]]
max_simil_history <- topics[[5]]
plot(silhouette_history)
colnames(words_similarity_matrix) <- selected_tokens
rownames(words_similarity_matrix) <- selected_tokens

# Adding names to topics
names(list_topics) <- paste0("Gr_", seq(length(list_topics)))

# Add info about Dunning statistics, sort topics and clean sentences
list_topics <- lambd_extractor(list_topics, articles_unnested, 
                               general_stats, min_counts = v_min_counts, 
                               min_lambda_daily = v_min_lambda_daily)
list_topics <- arrange_topics(list_topics, "max_lambda")
list_topics <- clear_sentences(list_topics)

#####################################################################
# Topics listing
# ###################################################################
# This enables to print out and check all topics
if(print_topics){
    for(name in names(list_topics)){
        print(list_topics[[name]][["max_lambda"]])
        print(list_topics[[name]][["words_DF"]])
        print("")
        print(paste0(list_topics[[name]][["site_name"]], ": ", list_topics[[name]][["sentences"]]))
        print("----------------------------------")
    }
}

#####################################################################
# Saving output for report
# ###################################################################
# Data frame with Dunning statistics for plotting
lambda_daily_DF <- calculate_lambda_statistics(articles_unnested, general_stats, 
                                               min_counts = v_min_counts, 
                                               min_lambda_daily = v_min_lambda_daily) %>% 
    mutate(lambda_log = log(lambda + 1) ) %>%
    rename(name = word) %>%
    dplyr::select(name, lambda, lambda_log)

# Saving topics list, similarity matrix and df with Dunning statistics
save(list_topics, 
     words_similarity_matrix, 
     lambda_daily_DF, 
     file = paste0(working_dir, "news-selectors/data/topics/daily_topics_list.RData"))


