# -*- coding: utf-8 -*-
from itertools import islice


class TextGenerationModel:

    def __init__(self, dp, sa, ts, refer):
        self.dp = dp
        self.sa = sa
        self.ts = ts
        self.refer = refer

    def get_templates(self, e):

        plans = self.dp.plan(e)

        plan_aggs = [(plan, self.sa.agg(plan['plan'])) for plan in plans]

        while plan_aggs:

            for plan, aggs in plan_aggs:

                try:
                    agg = next(aggs)
                except StopIteration:
                    plan_aggs.remove((plan, aggs))
                    continue

                selects = self.ts.select(agg['agg'], e)

                for select in selects:

                    result = {}
                    result.update(plan)
                    result.update(agg)
                    result.update(select)

                    yield result

    def get_n_templates(self, e, n):

        return list(islice(self.get_templates(e), 0, n))

    def get_best_template(self, e, n, ranking):

        return max(self.get_n_templates(e, n), key=ranking)

    def get_n_best_template(self, e, n1, n2, ranking):

        return list(sorted(self.get_n_templates(e, n1),
                           key=ranking,
                           reverse=True))[:n2]

    def get_n_best_texts(self, e, n1, n2, ranking):

        templates = self.get_n_best_template(e, n1, n2, ranking)

        for tt in templates:

            ctx = {'seen': set()}
            texts = [t.fill(a, self.refer, ctx)
                     for t, a in zip(tt['templates'], tt['agg'])]

            tt['text'] = ' '.join(texts)

        return templates

    def get_best_text(self, e, n=1, ranking=lambda x: 1):

        tt = self.get_best_template(e, n, ranking)

        ctx = {'seen': set()}
        texts = [t.fill(a, self.refer, ctx)
                 for t, a in zip(tt['templates'], tt['agg'])]

        return ' '.join(texts)

    def make_texts(self, X, n=1, ranking=lambda x: 1, outfilepath=None):

        with open(outfilepath, 'w', encoding='utf-8') as f:

            for i, e in enumerate(X):

                t = self.get_best_text(e, n, ranking)

                f.write('{}\n'.format(t))

                if i % 100 == 0:
                    print(i)
