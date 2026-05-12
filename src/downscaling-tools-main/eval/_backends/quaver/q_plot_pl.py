import os
import pandas as pd
import argparse
import matplotlib.cm as cm


######################
# q_plot_probabilistic
######################


list_ml_exp = {
    "is9i": "eefo O96->O320 (is9i)",
    "ioj2": "downscaling eefo O96->O320 (ioj2)",
    "j0ys": "j0ys ag 1e6",
}  ### list ml exps with additional information

list_colors = [
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ffff33",  # Yellow
    "#a65628",  # Brown
    "#f781bf",  # Pink
    "#999999",  # Gray
    "#66c2a5",  # Teal
]
nmem = 10
first_reference_date = 20230801
last_reference_date = 20230901
grid = "O320"

# parameters that do not change
class_ = "rd"
date_step = 24
first_lead_time = 24
last_lead_time = 240
lead_time_step = 24

database = "fdb"
type_plot = "sfc"

nmem_verif = nmem


def analysis_upperair(experiments, plots_results=None):
    import itertools

    vend = "an"

    parameters = ["t", "z", "u", "v"]
    levelist = [1000, 850, 500]
    domains = ["n.hem", "tropics", "s.hem"]

    all_scores_det = ["rmsef", "sdaf"]
    scores_mem = [p for p in all_scores_det if p not in "rmsef"]

    all_scores_ens = ["spread", "fcrps"]
    scores_pf = [p for p in all_scores_ens if p not in "spread"]

    to_plot_pf = []
    for param, level, domain, score in itertools.product(
        parameters, levelist, domains, scores_pf
    ):
        if not (param == "z" and level == 500 and domain == "tropics"):
            to_plot_pf.append((param, "pl", level, domain, score))

    to_plot_mem = []
    for param, level, domain, score in itertools.product(
        parameters, levelist, domains, scores_mem
    ):
        if not (param == "z" and level == 500 and domain == "tropics"):
            to_plot_mem.append((param, "pl", level, domain, score))

    to_plot_sp = []
    for param, level, domain in itertools.product(parameters, levelist, domains):
        if not (param == "z" and level == 500 and domain == "tropics"):
            to_plot_sp.append((param, "pl", level, domain))

    # scores for ensemble as a whole
    for p, t, l, d, s in to_plot_pf:

        param_lev = f"{p}{l}"
        if param_lev not in ("z500", "z1000", "t850", "ff850", "ff250"):
            continue

        models_list = [
            exp_curve(
                expver,
                "pf",
                s,
                vend,
                color=list_colors[i],
                add_info=add_info,
            )
            for i, (expver, add_info) in enumerate(experiments.items())
        ]
        models_list.append(ifs_curve("pf", s, vend))
        models_list.append(eefo_o96_an("pf", s, vend))

        plots_results.append(
            plot(
                kind="mean_plot",
                mean_method="fair",
                models=models_list,
                data=plotdata(
                    date=DateSequence(
                        first_reference_date, last_reference_date, date_step
                    ),
                    step=StepSequence(first_lead_time, last_lead_time, lead_time_step),
                    parameter=p,
                    levtype=t,
                    level=l,
                    domain=d,
                ),
                confidence_intervals="yes",
                confint="yes",
                title=[None, None, None, r"${mean_begin} to ${mean_end}"],
            )
        )

    # scores for ensemble spread/skill
    for p, t, l, d in to_plot_sp:

        param_lev = f"{p}{l}"
        if param_lev not in ("z500", "z1000", "t850", "ff850", "ff250"):
            continue

        models = []
        for expver, add_info in experiments.items():
            models.append(
                exp_curve(
                    expver,
                    "pf",
                    "spread",
                    vend,
                    suff="spread",
                    line_style="dashed",
                    add_info=add_info,
                )
                * (nmem / (nmem - 1)) ** 0.5
            )
            models.append(
                exp_curve(
                    expver,
                    "em",
                    "rmsef",
                    vend,
                    suff="error",
                    add_info=add_info,
                )
                * (nmem / (nmem + 1)) ** 0.5
            )

        plots_results.append(
            plot(
                kind="mean_plot",
                mean_method="fair",
                models=models,
                data=plotdata(
                    date=DateSequence(
                        first_reference_date, last_reference_date, date_step
                    ),
                    step=StepSequence(first_lead_time, last_lead_time, lead_time_step),
                    parameter=p,
                    levtype=t,
                    level=l,
                    domain=d,
                    score=s,
                ),
                confidence_intervals="yes",
                confint="yes",
                title=[None, None, None, r"${mean_begin} to ${mean_end}"],
            )
        )

    # scores for individual ensemble members
    for p, t, l, d, s in to_plot_mem:

        # ifs reference available only for a limited number of parameters/levels
        param_lev = f"{p}{l}"
        if param_lev not in ("z500", "z1000", "t850", "ff850", "ff250"):
            continue

        model_list = []
        for i in range(1, nmem_verif + 1):
            model_list.append(ifs_curve("pf", s, vend, suff=f"mem {i}", number=i))
        for i in range(1, nmem_verif + 1):
            for expid, add_info in experiments.items():
                model_list.append(
                    exp_curve(
                        expid,
                        "pf",
                        s,
                        vend,
                        suff=f"mem {i}",
                        number=i,
                        add_info=add_info,
                    )
                )

        plots_results.append(
            plot(
                kind="mean_plot",
                mean_method="fair",
                models=model_list,
                data=plotdata(
                    date=DateSequence(
                        first_reference_date, last_reference_date, date_step
                    ),
                    step=StepSequence(first_lead_time, last_lead_time, lead_time_step),
                    parameter=p,
                    levtype=t,
                    level=l,
                    domain=d,
                ),
                confidence_intervals="yes",
                confint="yes",
                title=[None, None, None, r"${mean_begin} to ${mean_end}"],
            )
        )


def exp_curve(
    expver,
    type_,
    score,
    vend,
    suff="",
    line_style="solid",
    color="black",
    add_info="",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream=f"prepml_{expver}_{vend}",
            Class="rd",
            stream="enfo",
            type=type_,
            expver=expver,
            score=score,
            **kargs,
        ),
        legend=f"{expver} %s  - downscaling - {add_info}" % suff,
        line_colour=color,
        line_style=line_style,
    )


def ifs_curve(
    type_,
    score,
    vend,
    stream="enfo",
    suff="",
    line_style="dashed",
    line_colour="blue",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream="oper_%s" % vend,
            Class="od",
            stream=stream,
            type=type_,
            expver="0001",
            score=score,
            **kargs,
        ),
        legend="IFS %s - target/truth" % suff,
        line_colour=line_colour,
        line_style=line_style,
    )


def eefo_o96(
    type_,
    score,
    vend,
    stream="eefo",
    suff="",
    line_style="dotted",
    line_colour="red",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream="qds_eefo_o96_ob",
            Class="od",
            stream="eefo",
            type=type_,
            expver="0001",
            score=score,
            **kargs,
        ),
        legend="eefo O96 %s - input" % suff,
        line_colour=line_colour,
        line_style=line_style,
    )


def eefo_o96_an(
    type_,
    score,
    vend,
    stream="eefo",
    suff="",
    line_style="dotted",
    line_colour="red",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream="qds_eefo_o96_an",
            Class="od",
            stream="eefo",
            type=type_,
            expver="0001",
            score=score,
            **kargs,
        ),
        legend="eefo O96 %s - input" % suff,
        line_colour=line_colour,
        line_style=line_style,
    )


def eefo_o320(
    type_,
    score,
    vend,
    stream="eefo",
    suff="",
    line_style="solid",
    line_colour="purple",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream="qds_eefo_o320_ob",
            Class="od",
            stream="eefo",
            type=type_,
            expver="0001",
            score=score,
            **kargs,
        ),
        legend="eefo O320 %s" % suff,
        line_colour=line_colour,
        line_style=line_style,
    )


def enfo_o320(
    type_,
    score,
    vend,
    stream="eefo",
    suff="",
    line_style="dashed",
    line_colour="orange",
    **kargs,
):
    return curve(
        data=modeldata(
            vstream="qds5_ob",
            Class="od",
            stream="enfo",
            type=type_,
            expver="0001",
            score=score,
            **kargs,
        ),
        legend="enfo O320 %s - target/truth" % suff,
        line_colour=line_colour,
        line_style=line_style,
    )


def ifs_curve_o1280(
    type_, score, vend, stream="enfo", suff="", line_style="dashed", **kargs
):
    return curve(
        data=modeldata(
            grid="O1280",
            vstream="oper_%s" % vend,
            Class="od",
            stream=stream,
            type=type_,
            expver="0001",
            score=score,
            # confintmaker="block_bootstrap",
            **kargs,
        ),
        legend="IFS %s O1280" % suff,
        line_colour="blue",
        line_style=line_style,
    )


def submit_plot():
    plots_results = []

    analysis_upperair(list_ml_exp, plots_results)

    document(
        plots=plots_results,
        data=documentdata(),
        orientation="landscape",
    )


submit_plot()
