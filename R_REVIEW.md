## MAG

[(length(na.omit(idx_gl))*n/60)](https://github.com/irinagain/iglu/blob/82e4d1a39901847881d5402d1ac61b3e678d2a5e/R/mag.R#L60) has to be 
```
    diffs = abs(diff(idx_gl))
    mag = sum(diffs, na.rm = TRUE)/
      (length(na.omit(diffs))*n/60)
```

## CGMS2DayByDay

[ndays = ceiling(as.double(difftime(max(tr), min(tr), units = "days")) + 1)](https://github.com/irinagain/iglu/blob/82e4d1a39901847881d5402d1ac61b3e678d2a5e/R/utils.R#L208) has to be ndays = ceiling(as.double(difftime(max(tr), min(tr), units = "days")))`


grid omits the first measurement of input data and shift timeline -dt0
[dti_cum = cumsum(dti)](https://github.com/irinagain/iglu/blob/82e4d1a39901847881d5402d1ac61b3e678d2a5e/R/utils.R#L210C13-L210C19) has to be `dti_cum = c(0,cumsum(dti))`

