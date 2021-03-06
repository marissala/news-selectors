# ###################################################################
# Biblioteki
# ###################################################################
install.packages("rmarkdown", repos='http://cran.us.r-project.org', lib="/home/app/R/x86_64-pc-linux-gnu-library/3.4")
library(rmarkdown)
library(lubridate)
library(magrittr)
#install.packages("mailR", repos='http://cran.us.r-project.org', dep = T)
#library(mailR)

install.packages("glue")
library(glue)

working_dir <- "/home/app/clustering/News_Clustering/" #"D:/Osobiste/GitHub/"
working_dir <- "~/Sentinel/"
# ###################################################################
# Generacja raportow
# ###################################################################
# Instrukcja - https://www.reddit.com/r/rstats/comments/7rpm5y/what_is_the_best_way_to_automatically_knit_an/

v_date <- Sys.time() %>% ymd_hms() %>% as.Date() - 1
rmarkdown::render(input = paste0(working_dir, 'news-selectors/scripts/reports/daily_report_html_en.Rmd'),
                  output_file = paste0(working_dir, 'news-selectors/data/daily_reports/Raport_dzienny_',v_date, '.html'), 
                  encoding = "UTF8")

rmarkdown::render(input = paste0(working_dir, 'news-selectors/scripts/reports/daily_report_word.Rmd'),
                  output_file = paste0(working_dir, 'news-selectors/data/daily_reports/Raport_dzienny_',v_date, '.docx'), 
                  encoding = "UTF8")

# ###################################################################
# Wysyłanie maila
# ###################################################################
#from <- "sender@example.com"
#to <- c("receiver@example.com")
#subject <- paste0("Raport dzienny ", v_date)
#fileName <- c(paste0(working_dir, 'news-selectors/data/daily_reports/Raport_dzienny_', v_date, '.html'),
#              paste0(working_dir, 'news-selectors/data/daily_reports/Raport_dzienny_', v_date, '.docx'))
#body <- paste0("To jest mail z raportem całodziennym z ", v_date)
#
#send.mail(from = from,
#          to = to,
#          subject = subject,
#          attach.files = fileName,
#          html = T,
#          inline = T,
#          body = body,
#          smtp = list(host.name = "host_name", port = 465, user.name = "sender@example", passwd = "sender_password", ssl = TRUE),
#          authenticate = TRUE,
#          send = TRUE
#)
