import sys

with open("walk_forward.py", encoding="utf-8") as f:
    content = f.read()

repl1 = """    overwrite: bool = False,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
):"""
sub1 = """    overwrite: bool = False,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
    sl_scores_dir: str | None = None,
):"""
content = content.replace(repl1, sub1)

repl2 = """    temporal_extractor: bool = False,
    candidate_pairs: list[tuple[str, bool]] | None = None,
):"""
sub2 = """    temporal_extractor: bool = False,
    candidate_pairs: list[tuple[str, bool]] | None = None,
    sl_scores_dir: str | None = None,
):"""
content = content.replace(repl2, sub2)

repl3 = """def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward research validation")"""
sub3 = """def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward research validation")
    parser.add_argument("--sl-scores-dir", type=str, default=None, help="Path to directory containing sl_scores_{period}_h5.csv for S5 Integration")"""
content = content.replace(repl3, sub3)

repl4 = """    if args.candidates:
        run_candidate_set(
            timesteps=timesteps,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
        )"""
sub4 = """    if args.candidates:
        run_candidate_set(
            timesteps=timesteps,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
            sl_scores_dir=args.sl_scores_dir,
        )"""
content = content.replace(repl4, sub4)

repl5 = """        run_research_matrix(
            timesteps=timesteps,
            algos=algos,
            cash_modes=cash_modes,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
        )"""
sub5 = """        run_research_matrix(
            timesteps=timesteps,
            algos=algos,
            cash_modes=cash_modes,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
            sl_scores_dir=args.sl_scores_dir,
        )"""
content = content.replace(repl5, sub5)

with open("walk_forward.py", "w", encoding="utf-8") as f:
    f.write(content)
