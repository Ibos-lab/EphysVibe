class TrialsData:
    def __init__(
        self,
        sp_samples,
        blocks,
        code_numbers,
        code_samples,
        eyes_values,
        lfp_values,
        samples,
        clusters_id,
        clusters_ch,
        clustersgroup,
        clusterdepth,
    ):
        self.sp_samples = (sp_samples,)
        self.blocks = (blocks,)
        self.code_numbers = (code_numbers,)
        self.code_samples = (code_samples,)
        self.eyes_values = eyes_values
        self.lfp_values = (lfp_values,)
        self.samples = (samples,)
        self.clusters_id = (clusters_id,)
        self.clusters_ch = (clusters_ch,)
        self.clustersgroup = (clustersgroup,)
        self.clusterdepth = (clusterdepth,)
        self.check_shapes()

    def check_shapes(self):
        self.sp_samples
