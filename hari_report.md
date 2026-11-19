Figure 1 (insert image: depth_by_class.png)

Caption: False positives produce a wider range of pitch drops than
confirmed planets, including many very deep drops, while both groups
overlap at shallow drops.

There are two categorical groups which are Nasa's false positive planets and the confirmed planets. We are comparing the pitch drop (in semitones) of the 50 stars with detected dips in both groups using a box plot to find if there is overlap in range and how the drops vary between the groups. The graph shows that false positives tend to have larger range of deep drops compared to the confirmed planets. It is important to consider that there is also a large overlap in the ranges at the bottom of both groups as well, showing at smaller drops there is overlap between groups. Using a box plot allows us to see the median of both groups to compare how spread out the data is.  


Figure 2 (inset image: noise_vs_depth)

Caption: Noise (in Hz) and pitch drop (in semitones) of false positives and confirmed
planets. Confirmed planets tend to cluster at lower noise with similar pitch
drops, whereas false positives are more spread out, with deeper pitch drops
and higher noise. Both groups overlap at low noise.

This graph features a scatter plot comparing two groups of data (false positives and confirmed planets) by their noise (x-axis) and their pitch drop depth (y-axis), for the 50 stars with detected dips. The scatter plot allows us to see whether there is any relationship between the continuous variables of noise and pitch drop for the two groups. The data shows that confirmed planets tend to cluster around lower noise and pitch drops, whereas the false positives are more spread out and vary in higher noise and pitch drop depth. For example, 13 of the 15 stars with noise above 5 Hz are false positives. However, a few of the false positives did cluster with the confirmed planets at lower noise and depth. The legend shows the two groups: confirmed planets in blue and false positives in orange. It is important to note this data is based on a subset of 50 stars and folding each star's light curve on its orbital period.


Size of data: 50 rows, 8 Columns
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

There is no missing data.
The columns that answer my research question are: pitch_drop_semitones, noise_hz, dip_to_noise, koi_disposition


What tasks need to be completed and how much time: I still need to use a statistical test that uses a t-test for validation of the results. I also need to work on reviewing the code to make sure I have a complete understanding the flow of the code so I can get ready to start on the presentation aspect of the project. I have a lot of flake8 problems and functions with no docstrings, I plan to work on that last after writing code and EDA. I estimate around 3-4 more hours spent on these tasks, could be more. 

8. Work plan evaluation: I took more time than what was expected in the work plan, ended up having around 10-12 hours of work in total. I believe this maybe due the complexity of this project as it also required me to find new and different methods I haven't used before in this code. There were many steps to this project, and working on implementing everything I learned into one project was a little overwhelming but fun, as it showed me how the methods I learned are used in real life scenarios. It also required me to look through the internet for documentation and different methods I haven't used before in this code and especially the pandas methods for the data analysis.

10. Collaboration: I have used the documentation for matplotlib, pandas, numpy, and OS. This was for finding how to use specific functions and guides for data analysis. I also used claude for explaining some of the functions and asked it to guide
me in finding the best functions in the libaries for a specific task. I also used claude for explaining reference code to see how I could implement it into my project and it also assisted me in planning the project out, such as 
splitting the work evenly over the course a few weeks and creating a schedule to get tasks done for this project. When I had errors in my code I also allowed AI review what went wrong and implemented its feedback to create working code.


Q3: Can sound still tell real planets from false alarms?

Null Hypothesis: The sound model and brightness model do equally well in differentiating between confirmed planets or false positives.

Alternative Hypothesis: The sound model and brightness model do have a difference in how they perform in differentiating between confirmed planets and false positives.

