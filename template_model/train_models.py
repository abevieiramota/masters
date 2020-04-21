from pretrained_models import *

# ds = ('train', 'dev')

# _ = make_dp_lm(ds, 2)
# _ = make_dp_lm(ds, 3)
# _ = make_dp_lm(ds, 4)

# _ = make_sa_lm(ds, 2)
# _ = make_sa_lm(ds, 3)
# _ = make_sa_lm(ds, 4)

# _ = make_main_model_data(ds) 

# _ = make_pretrained_abe_ref_dbs(ds[0])
# _ = make_pretrained_abe_ref_dbs(ds[1])
# _ = make_reg_lm(ds, 3)
# _ = make_reg_lm(ds, 4)

# _ = make_template_db(ds)

# _ = make_template_selection_lm(ds, 3, 'lower')
# _ = make_template_selection_lm(ds, 4, 'lower')
# _ = make_text_selection_lm(ds, 3, 'lower')
# _ = make_text_selection_lm(ds, 4, 'lower')

ds = ('train', 'dev')

_ = make_dp_lm(ds, 2)
_ = make_dp_lm(ds, 3)
_ = make_dp_lm(ds, 4)

_ = make_sa_lm(ds, 2)
_ = make_sa_lm(ds, 3)
_ = make_sa_lm(ds, 4)

_ = make_template_db(ds)
make_template_selection_lm(ds, 3, 'markov')
make_template_selection_lm(ds, 4, 'markov')
make_template_selection_lm(ds, 5, 'markov')

_ = make_pretrained_abe_ref_dbs(ds[0])
_ = make_pretrained_abe_ref_dbs(ds[1])
_ = make_reg_lm(ds, 3)
_ = make_reg_lm(ds, 4)
_ = make_reg_lm(ds, 5)

_ = make_text_selection_lm(ds, 3, 'markov')
_ = make_text_selection_lm(ds, 4, 'markov')
_ = make_text_selection_lm(ds, 5, 'markov')
