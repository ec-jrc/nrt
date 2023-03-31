from nrt.monitor import BaseNrt


class bayts(BaseNrt):
    def __init__(self, pdfs, chi=0.9, mask=None, **kwargs):
        

    def monitor(self, array, date):
        # TODO: Use typing to allow numpy array, DataArray or Dataset for the
        # array argument.
        # If array, must be 2D and pdfs must contain a single pair of distributions
        # If DataArray, only one time step at the time is allowed; in case of multiple
        # pdfs provided during instantiation, DataArray name must be non None
        # If Dataset, same, only one time step
        if not isinstance(date, datetime.date):
            raise TypeError("'date' has to be of type datetime.date")
        if self.detection_date is None:
            self.detection_date = np.zeros_like(self.mask, dtype=np.uint16)
        # Compute a mask of values that can be worked on
        is_valid = np.logical_and(self.mask == 1, np.isfinite(array))
        # Compute PF and PNF
        pnf = np.where(is_valid, self.pdfs['pnf'].pdf(array), np.nan)
        pf = np.where(is_valid, self.pdfs['pf'].pdf(array), np.nan)
        # Bring small PNF values back to zero (PNF = 0 if PNF < 1e-10000 else PNF)
        pnf[pnf < 1e-10000] = 0
        # Compute PNF_cond (PNF/(PF + PNF))
        pnf_cond (pnf/(pf + pnf))
        # Clip PNF_cond values to the 0.1,0.9 interval
        pnf_cond = np.clip(pnf_cond, 0.1, 0.9)
        # Logic to decide whether and how to update p_change (based on previous
        # value of p_change and current PNF_cond value
        # Create an intermediary boolean array indicating pixels to be updated
        process_update = np.logical_or(self.process >= 0.5, pnf_cond >= 0.5)
        # Two update strategies depending only on whether process is zero or not
        process_new = np.where(self.process == 0,
                               self.bayesian_update(self.pnf_cond_before, pnf_cond),
                               self.bayesian_update(self.process, pnf_cond))
        process_new[~process_update] =0
        # If process_new is below 0.5 while process was below 0.5, set process_new
        # to zero (unflagging)
        process_new = np.where(np.logical_and(self.process >= 0.5, process_new < 0.5),
                               0, process_new)
        # Replace self.process by process_new
        self.process = process_new
        # Update mask (3 value corresponds to a confirmed break)
        is_break = self.process > self.chi
        to_update = np.logical_and(is_valid, is_break)
        self.mask[to_update] = 3
        # Update detection date
        days_since_epoch = (date - datetime.datetime(1970, 1, 1)).days
        self.detection_date[to_update] = days_since_epoch
        # Store pnf_cond for t+1
        self.pnf_cond_before = pnf_cond


    @staticmethod
    def bayesian_update(p_change, pnf_cond):
            return (p_change * pnf_cond)/((p_change * pnf_cond) + ((1 - p_change) * (1 - pnf_cond)))
