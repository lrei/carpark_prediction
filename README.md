Predicting The Availability of Parking Spaces in Ljubljana
==================

Predicting the availability of parking spaces in Ljubljana car parks.
Time series (well, time window) regression using linear regression, regression trees and random forest.
Predicting for 30min, 1h, 2h and 3h intervals.

The data can be collected from [Open Data Slovenia](http://opendata.si). Doubt I can share the dataset that was provided to me but you can always try collecting a new one.

* Machine learning part comes from [Scikit-learn](http://scikit-learn.org).
* Uses [Pandas](http://pandas.pydata.org/) for data manipulation/transformation e.g. resampling.
* Plotting using Pandas / [Pyplot](http://matplotlib.org)

The particular dataset that was provided to me consisted of a set of gzipped json files and required a lot of cleanup. I initially imported it into an sqlite database and did much of the cleanup there. The code for this is in the source code but I imagine it is next to useless for anyone else.


Additional materials:
[Report](http://www.scribd.com/doc/197263647/PREDICTING-THE-AVAILABILITY-OF-PARKING-SPACES-IN-LJUBLJANA)
[Presentation slides](http://www.scribd.com/doc/197262974/Predicting-The-Availability-of-Parking-Spaces-in-Ljubljana)


Notes: 
This was done as part of a course assignment for the Josef Stefan International
Postgraduate School.

This is mostly an ugly version of a iPython Notebook since I was using tmux with
vim and ipython panes... the code is uncommented and ugly but it is complementary
to both the report and the presentation and might be useful if you plan on doing
something similar.
