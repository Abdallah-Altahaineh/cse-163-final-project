Size of data: 50 rows, 10 Columns
Each row is one kepler star
Columns: 
- baseline_hz: the star's normal pitch outside the dip (Hz)
- pitch_drop_hz: pitch drop in Hz
- pitch_drop_semitones: pitch drop in semitones
- noise_hz: the stars pitch moving outside the dips (Hz)
-dip_to_noise: pitch_drop_hz/noise_hz to show if dip stands out, a higher ratio shows a clearer dip
- audible: whether the pitch drop is big enough to hear (at least 12 cents)
- kepid: unique id from NASA for Kepler star
-koi_disposition: NASA's own label for whether the star was a confirmed planet or a false positive 
- duration_hr: folded dip length in hours, measured by half its depth
-flatness: avg drop inside the dip/deepest drop, describing the shape of the bottom of the dip. If its flat U shape, its close to 1, most of the dip is at this level. If its more V shaped, average is less than deepest, level around .75. Deepest value measures one point, average includes all other points in dip


There is missing data for duration_hr and flatness. This is due to the minimum threshold set for stars that are too faint dips and long periods of dips with fewer than 15 folded slices. They are set as NaN, total of 14 for duration hour and 26 for flatness. 


The columns that answer my research question are: pitch_drop_semitones, noise_hz, dip_to_noise, koi_disposition, duration_hr, and flatness. These connect to the research question of: can sound tell confirmed planets apart from false positives, because when measuring the you need to look at how much the drop is in semitones to consider it as a dip. noise_hz allows for checking the noise outside the dip to see how easy it is to hear it over the noise. And the koi_disposition allows us to compare the two groups. Dip to noise checks how many times bigger the drop is to the noise, the larger the number the more the clear the dip is. I also used duration_hr to show how long the the dorp was. The flatness shows the shape of the drop (U or V) and allows us to compare the avg to the lowest drop point. Causing a longer lasting a drop or a shorter V shaped drop. These features allow to tell apart a false positive and a confirmed planet. 


Summary: 
Seven-number summary (On 50 stars):        

       pitch_drop_semitones  noise_hz  dip_to_noise  duration_hr  flatness
count                 50.00     50.00         50.00        36.00     24.00
mean                   4.46     11.17         16.31         4.91      0.80
std                    8.38     35.04         27.62         9.73      0.06
min                    0.04      0.23          1.85         0.70      0.68
25%                    0.26      0.65          4.80         1.59      0.75
50%                    0.56      1.53          8.29         2.44      0.79
75%                    2.07      8.13         14.37         4.21      0.83
max                   27.98    239.38        143.92        57.92      0.90

koi_disposition
FALSE POSITIVE    25
CONFIRMED         25

Median of each feature, by group:
                 pitch_drop_semitones  noise_hz  dip_to_noise  duration_hr  flatness
koi_disposition                                                                     
CONFIRMED                        0.41      0.84          9.86         2.58      0.80
FALSE POSITIVE                   1.62      8.63          6.20         2.10      0.79


Differ between groups: The results show that the false positives tend to be much noiser (8.63hz for false positives and 0.84Hz for confirmed). The dips are also less clear for the false positives as the dip to noise ratio for the false positives are 6.197 compared to the dip to noise ratio of 9.856 for the confirmed. False positives pitch drop also tend to have a higher pitch drop of 1.622 semitones compared to 0.40998 semitones for the confirmed. 


Figure 1 (insert image: depth_by_class.png)

Caption: False positives produce a wider range of pitch drops than
confirmed planets, including many very deep drops, while both groups
overlap at shallow drops.

There are two categorical groups which are Nasa's false positive planets and the confirmed planets. We are comparing the pitch drop (in semitones) of all 50 stars in both groups using a box plot to find if there is overlap in range and how the drops vary between the groups. The graph shows that false positives tend to have larger range of deep drops compared to the confirmed planets. It is important to consider that there is also a large overlap in the ranges at the bottom of both groups as well, showing at smaller drops there is overlap between groups. Using a box plot allows us to see the median of both groups to compare how spread out the data is.  


Figure 2 (insert image: noise_vs_depth.png)
.png
Caption: Noise (in Hz) and pitch drop (in semitones) of false positives and confirmed
planets. Confirmed planets tend to cluster at lower noise with similar pitch
drops, whereas false positives are more spread out, with deeper pitch drops
and higher noise. Both groups overlap at low noise.

This graph features a scatter plot comparing two groups of data (false positives and confirmed planets) by their noise (x-axis) and their pitch drop depth (y-axis), for the 50 stars. The scatter plot allows us to see whether there is any relationship between the continuous variables of noise and pitch drop for the two groups. The data shows that confirmed planets tend to cluster around lower noise and pitch drops, whereas the false positives are more spread out and vary in higher noise and pitch drop depth. For example, 13 of the 15 stars with noise above 5 Hz are false positives. However, a few of the false positives did cluster with the confirmed planets at lower noise and depth. The legend shows the two groups: confirmed planets in blue and false positives in orange. It is important to note this data is based on a subset of 50 stars and folding each star's light curve on its orbital period.


Result validity: 

Q3: Can sound still tell real planets from false alarms?

Null Hypothesis: The sound model and brightness model do equally well in differentiating between confirmed planets and false positives.

Alternative Hypothesis: The sound model and brightness model do have a difference in how they perform in differentiating between confirmed planets and false positives.


Challenge goals: I went beyond the basic analysis by incorperating functions and methods not discussed in class that were in some of the libraries documentation. I also worked with messy Kepler light data and cleaned them up. 


8. Work plan evaluation: I took more time than what was expected in the work plan, ended up having around 10-12 hours of work in total. I believe this maybe due the complexity of this project as it also required me to find new and different methods I haven't used before in this code. There were many steps to this project, and working on implementing everything I learned into one project was a little overwhelming but fun, as it showed me how the methods I learned are used in real life scenarios. It also required me to look through the internet for documentation and different methods I haven't used before in this code and especially the pandas methods for the data analysis.

What tasks need to be completed and how much time: I still need to is train a classifier on audio and brightness to predict if the star is a confirmed or false positive. I also need to work on reviewing the code to make sure I have a complete understanding the flow of the code so I can get ready to start on the presentation aspect of the project. I plan to work on flake8 and dosctrings last after writing code and EDA. I estimate around 8-10 more hours spent on these tasks, could be more. 


Testing: I used assert tests on fake stars for 0.5% deep every 10 days, the function found pitch drop of about 5 semitones. Using a fake flat star, the pitch drop must stay under 0.5 semitones. Using the real 50 stars I found no repeated stars. I used fake stars to compare the expected to the observed. I also checked against real data, my results showed the fake stars passing, but the method on real stars showed most the dips were just spikes in the noise. For this, I switched to folding which fixed it. I also looked at the graphs to visualize what was happening and if the results made sense. All tests passed in the testing file. 


10. Collaboration: I have used the documentation for all libraries used such as matplotlib, pandas, numpy, and OS. This was for finding how to use specific functions and guides for data analysis. I also used claude for explaining some of the functions and asked it to guide
me in finding the best functions in the libaries for a specific task. I also used claude for explaining reference code to see how I could implement it into my project and it also assisted me in planning the project out, such as 
splitting the work evenly over the course a few weeks and creating a schedule to get tasks done for this project. When I had errors in my code I also allowed AI review what went wrong and implemented its feedback to create working code. I used abdullahs functions for folding and he assisted me in explaining me how they work for my own use and in my code. He also reviewed my code for feedback to implement.