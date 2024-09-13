import json
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    AGGS = [
        "prod",
        "min",
        "max",
        "mean",
        "sum_logit",
        "mean_logit",
        "mean_odd",
        "sum_odd",
    ]

    COLORS = [
        "#d62728",
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    for M1, M2 in [("7b", "7b"), ("34b", "34b")]:
        for scheme, scheme_name in [
            ("weighted_accs", "Weighted Voting"),
            ("best_of_n_accs", "Best-of-N"),
        ]:
            name_prefix = f"SFT-{M1} + PRM-{M2} + {scheme_name}"
            output_filename = f"/nobackup/users/yikangs/zhiqings/math/auto_figs/final_{scheme}_{M1}_{M2}_agg.pdf"
            accs = {}

            for AGG in AGGS:
                file_name = f"/nobackup/users/yikangs/zhiqings/math/figs/{scheme}_{M1}_{M2}_{AGG}_0.json"
                with open(file_name, "r") as f:
                    acc = json.load(f)[1:]
                accs[AGG] = acc

            set_of_voting_nums = set()
            for item in accs[AGG]:
                set_of_voting_nums.add(item["voting_num"])
            set_of_voting_nums = sorted(list(set_of_voting_nums))

            for acc_key in ["Level1-3", "Level4-5", "Level1-5"]:
                x = []
                y_accs = {}
                y_acc_vars = {}
                for voting_num in set_of_voting_nums:
                    x.append(voting_num)
                    # for accs, y_acc, y_acc_var in zip(
                    #     [majority_accs, weighted_accs, best_of_n_accs],
                    #     [y1_acc, y2_acc, y3_acc],
                    #     [y1_acc_var, y2_acc_var, y3_acc_var],
                    # ):
                    #     y = np.array(
                    #         [item[acc_key] for item in accs if item["voting_num"] == voting_num]
                    #     )
                    #     y_acc.append(y.mean())
                    #     y_acc_var.append(y.std())
                    for i in range(len(AGGS)):
                        y = np.array(
                            [
                                item[acc_key]
                                for item in accs[AGGS[i]]
                                if item["voting_num"] == voting_num
                            ]
                        )
                        if AGGS[i] not in y_accs:
                            y_accs[AGGS[i]] = []
                            y_acc_vars[AGGS[i]] = []
                        y_accs[AGGS[i]].append(y.mean())
                        y_acc_vars[AGGS[i]].append(y.std())

                del x[-1]
                # y1_acc = y1_acc[: len(x)]
                # y2_acc = y2_acc[: len(x)]
                # y3_acc = y3_acc[: len(x)]
                # y1_acc_var = y1_acc_var[: len(x)]
                # y2_acc_var = y2_acc_var[: len(x)]
                # y3_acc_var = y3_acc_var[: len(x)]
                for AGG in AGGS:
                    y_accs[AGG] = y_accs[AGG][: len(x)]
                    y_acc_vars[AGG] = y_acc_vars[AGG][: len(x)]

                x = np.array(x)
                # y1_acc = np.array(y1_acc) * 100
                # y2_acc = np.array(y2_acc) * 100
                # y3_acc = np.array(y3_acc) * 100
                # y1_acc_var = np.array(y1_acc_var) * 100  # Convert to percentage
                # y2_acc_var = np.array(y2_acc_var) * 100  # Convert to percentage
                # y3_acc_var = np.array(y3_acc_var) * 100  # Convert to percentage
                for AGG in AGGS:
                    y_accs[AGG] = np.array(y_accs[AGG]) * 100
                    y_acc_vars[AGG] = np.array(y_acc_vars[AGG]) * 100

                plt.clf()
                fig, ax1 = plt.subplots(figsize=(8, 6))

                # plot the file name
                if acc_key == "Level1-3":
                    fig_name = f"{name_prefix}\nAccuracy on Easy (Level 1-3) Problems"
                elif acc_key == "Level4-5":
                    fig_name = f"{name_prefix}\nAccuracy on Hard (Level 4-5) Problems"
                else:
                    fig_name = f"{name_prefix}\nAccuracy on All (Level 1-5) Problems"

                fontsize = 20
                legend_fontsize = 20
                tick_fontsize = 20

                fig.suptitle(
                    fig_name,
                    fontsize=fontsize,
                    horizontalalignment="center",  # Ensure the title is centered
                )

                # for color, label, y_acc, y_acc_var in zip(
                #     ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"],
                #     ["Majority Voting", "Weighted Voting w/ RM", "Best-of-N w/ RM"],
                #     [y1_acc, y2_acc, y3_acc],
                #     [y1_acc_var, y2_acc_var, y3_acc_var],
                # ):
                for i in range(len(AGGS)):
                    color = COLORS[i]
                    label = AGGS[i]
                    y_acc = y_accs[AGGS[i]]
                    y_acc_var = y_acc_vars[AGGS[i]]

                    # plot and fill_between
                    ax1.plot(x, y_acc, "-o", color=color, label=label, markersize=4)
                    # ax1.fill_between(
                    #     x,
                    #     y_acc - y_acc_var,
                    #     y_acc + y_acc_var,
                    #     alpha=0.2,
                    #     color=color,
                    # )

                plt.xscale("log")
                desired_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                desired_ticks = [tick for tick in desired_ticks if tick <= x.max()]

                plt.xticks(
                    desired_ticks,
                    labels=[str(tick) for tick in desired_ticks],
                    fontsize=tick_fontsize,
                )
                plt.yticks(fontsize=fontsize)
                plt.xlabel("N = number of solutions per problem", fontsize=fontsize)
                plt.ylabel("% Problems Solved", fontsize=fontsize)
                plt.legend(fontsize=legend_fontsize)
                plt.grid(False)
                plt.tight_layout()
                plt.savefig(
                    output_filename.replace(".pdf", f"_{acc_key}.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                )
