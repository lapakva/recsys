class Search:
    def __init__(self) -> None:
        pass

    def match(self, talent: dict, job: dict) -> dict:
        # ==> Method description <==
        # This method takes a talent and job as input and uses the machine learning
        # model to predict the label. Together with a calculated score, the dictionary
        # returned has the following schema:
        #
        # {
        #   "talent": ...,
        #   "job": ...,
        #   "label": ...,
        #   "score": ...
        # }
        #
        pass

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        # ==> Method description <==
        # This method takes a multiple talents and jobs as input and uses the machine
        # learning model to predict the label for each combination. Together with a
        # calculated score, the list returned (sorted descending by score!) has the
        # following schema:
        #
        # [
        #   {
        #     "talent": ...,
        #     "job": ...,
        #     "label": ...,
        #     "score": ...
        #   },
        #   {
        #     "talent": ...,
        #     "job": ...,
        #     "label": ...,
        #     "score": ...
        #   },
        #   ...
        # ]
        #
        pass
