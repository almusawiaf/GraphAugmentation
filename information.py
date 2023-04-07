files = ['Lung','Prostate']
outcome = ['Pass', 'Fail', 'Excluded']
    
DB_lung = ["temp-lung-dvh.csv",
            "washu_measures_lung_all.csv",
            "washu_measures_lung_all_dvh_flag.csv",
            "washu_measures_lung_all_dvh_value.csv"]


DB_prostate = ["temp-prostate-dvh.csv",
                "washu_measures_prostate_all.csv",
                "washu_measures_prostate_all_dvh_flag.csv",
                "washu_measures_prostate_all_dvh_value.csv"]


Prostate_QMs = ['vha_id','QualityMeasure1','QualityMeasure10','QualityMeasure11','QualityMeasure12','QualityMeasure13','QualityMeasure14','QualityMeasure15','QualityMeasure15_color','QualityMeasure16','QualityMeasure17A','QualityMeasure17B','QualityMeasure18','QualityMeasure19','QualityMeasure2','QualityMeasure24','QualityMeasure3','QualityMeasure4','QualityMeasure5','QualityMeasure6','QualityMeasure7','QualityMeasure8','QualityMeasure9']
Lung_QMs = ['vha_id','QualityMeasure1','QualityMeasure10','QualityMeasure11','QualityMeasure12','QualityMeasure13','QualityMeasure14','QualityMeasure15','QualityMeasure15Chemo','QualityMeasure15RT','QualityMeasure15Surgery','QualityMeasure16','QualityMeasure17','QualityMeasure18','QualityMeasure19','QualityMeasure19_color','QualityMeasure2','QualityMeasure20','QualityMeasure21A','QualityMeasure21B','QualityMeasure22','QualityMeasure23','QualityMeasure24','QualityMeasure27','QualityMeasure3','QualityMeasure4','QualityMeasure5','QualityMeasure6','QualityMeasure7','QualityMeasure8A','QualityMeasure8B','QualityMeasure9']

distance = ['euclidean']
#,'minkowski','cityblock','seuclidean','sqeuclidean','cosine','correlation','hamming','jensenshannon','chebyshev','canberra','braycurtis','mahalanobis','yule','matching','dice','kulczynski1','rogerstanimoto','russellrao','sokalmichener']

QM = [Lung_QMs, Prostate_QMs]

ALL = [DB_lung, DB_prostate]
