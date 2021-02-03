import numpy as np
import nrt.cusum as cs


def test_history_roc(X_y_dates_romania):
    """Validated against roc_history() in the R package bFast
    res_2d <- apply(y, 2, function(column){
      # convert to dataframe
      X_df <- as.data.frame(X)
      X_df$y <- column
      #Remove nan
      X_df_clear <- X_df[!is.na(X_df$y),]

      level <- 0.05

      n <- nrow(X_df_clear)
      data_rev <- X_df_clear[n:1, ]
      y_rcus <- efp(y ~ V1+V2+V3+V4+V5, data = data_rev, type = "Rec-CUSUM")
      y_start <- if (sctest(y_rcus)$p.value < level) {
        length(y_rcus$process) - min(which(abs(y_rcus$process)[-1] >
                                             boundary(y_rcus)[-1])) + 1
      } else {
        1
      }
      return(y_start)
    })
    """
    X, y, dates = X_y_dates_romania
    result = np.array([1, 8, 49, 62, 1], dtype='float32')
    stable_idx = np.zeros(y.shape[1])
    for idx in range(y.shape[1]):
        # subset and remove nan
        is_nan = np.isnan(y[:, idx])
        _y = y[~is_nan, idx]
        _X = X[~is_nan, :]

        # get the index where the stable period starts
        stable_idx[idx] = cs.history_roc(_X, _y)

    # Result from strucchange must be subtracted by 1, because R is 1 indexed
    np.testing.assert_allclose(stable_idx, result-1)


def test_efp(X_y_dates_romania, strcchng_efp):
    X, y, dates = X_y_dates_romania

    is_nan = np.isnan(y[:, 0])
    _y = y[~is_nan, 0]
    _X = X[~is_nan, :]

    process = cs._cusum_rec_efp(_X[::-1], _y[::-1])

    result = strcchng_efp

    # Relative high tolerance, due to floating point precision
    np.testing.assert_allclose(process[X.shape[1]+2:], result[X.shape[1]+2:],
                               rtol=1e-02)

