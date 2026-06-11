import os
import pandas as pd
import argparse
import matplotlib.cm as cm


######################
# q_plot_probabilistic
######################


list_ml_exp = {
    "iytd": "5k steps / 100k",
    "iysd": "13k steps / 100k",
    "iytc": "27k steps / 100k",
    "iz2p": "50k steps / 100k",
    "iz2o": "100k steps / 100k",
}  ### list ml exps with additional information
list_ml_exp = {
    "ip6y": "ip6y original after ft",
    "ioj2": "ioj2 original 1e6",
    "j0ys": "j0ys ag 1e6",
    "j10e": "j10e ag 1e6",
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


def observations_surface(experiments, plots_results):
    import itertools

    vend = "ob"

    all_parameters = ["2t", "10ff", "2d"]
    parameters = [p for p in all_parameters if p not in ["tp"]]
    domains = ["n.hem"]

    all_scores_det = ["seeps"]
    scores_mem = [p for p in all_scores_det if p not in "seeps"]

    all_scores_ens = ["fcrps", "spread"]
    scores_pf = [p for p in all_scores_ens]  # if p not in "spread"]

    to_plot_pf = []
    for param, domain, score in itertools.product(all_parameters, domains, scores_pf):
        to_plot_pf.append((param, 0, domain, score))

    to_plot_mem = []
    for param, domain, score in itertools.product(parameters, domains, scores_mem):
        to_plot_mem.append((param, 0, domain, score))

    if "tp" in all_parameters and "seeps" in all_scores_det:
        for domain in domains:
            to_plot_mem.append(("tp", 24, domain, "seeps"))

    # scores for ensemble as a whole
    for p, a, d, s in to_plot_pf:
        models = [
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
        models.append(enfo_o320("pf", s, vend))
        models.append(eefo_o96("pf", s, vend))

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
                    period=a,
                    levtype="sfc",
                    domain=d,
                    climat="rodw",
                ),
                confidence_intervals="yes",
                confint="yes",
                title=[None, None, None, r"${mean_begin} to ${mean_end}"],
            )
        )

    # scores for individual ensemble members
    for p, a, d, s in to_plot_mem:
        for expver, (add_info, color) in experiments.items():
            model_list = [ifs_curve("fc", s, vend, stream="oper", suff=f"ctrl")]
            for i in range(1, nmem_verif + 1):
                model_list.append(
                    exp_curve(expver, "pf", s, vend, suff=f"mem {i}", number=i)
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
                        step=StepSequence(
                            first_lead_time, last_lead_time, lead_time_step
                        ),
                        parameter=p,
                        period=a,
                        levtype="sfc",
                        domain=d,
                        climat="rodw",
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
        legend=f"{add_info}",
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
        legend="IFS %s" % suff,
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
        legend="eefo O96 %s forecast" % suff,
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
        legend="eefo O96 %s forecast" % suff,
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
    stream="enfo",
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
        legend="enfo O320 %s forecast" % suff,
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

    observations_surface(list_ml_exp, plots_results)

    document(
        plots=plots_results,
        data=documentdata(),
        orientation="landscape",
    )


submit_plot()
