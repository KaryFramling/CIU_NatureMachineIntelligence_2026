# NMIFigures.R
# Figure generation script for the Nature Machine Intelligence paper:
# "Re-founding Explainable AI: Contextual Importance and Utility
#  as a Universal Explanation Framework"
#
# Uses the master branch of https://github.com/KaryFramling/ciu
# Datasets: German Credit (rchallenge package), Ames Housing (AmesHousing)
#
# Output: figures/*.pdf  (cairo_pdf, Nature column widths)
# Cache:  cache/*.rds    (expensive model training / CIU scans)
#
# Run from the empirical/ directory:
#   setwd("empirical"); source("NMIFigures.R"); make.all.figures()

# ── Dependencies ─────────────────────────────────────────────────────────────
library(ciu)
library(ggplot2)
library(patchwork)   # side-by-side panels
library(caret)
library(rchallenge)  # German Credit
library(AmesHousing) # Ames Housing
library(igraph)

# ── Output directories ────────────────────────────────────────────────────────
dir.create("figures", recursive = TRUE, showWarnings = FALSE)
dir.create("cache",   recursive = TRUE, showWarnings = FALSE)

# ── Nature figure dimensions (inches) ────────────────────────────────────────
FIG_W_1COL <- 3.50
FIG_W_2COL <- 7.20
FIG_H      <- 4.20

# Save helper – cairo_pdf for production-quality vector output
save.fig <- function(p, name, width = FIG_W_2COL, height = FIG_H) {
  path <- file.path("figures", paste0(name, ".pdf"))
  cairo_pdf(path, width = width, height = height)
  print(p)
  dev.off()
  invisible(path)
}

# Cache helpers
cache.save <- function(obj, name) {
  saveRDS(obj, file.path("cache", paste0(name, ".rds")))
}
cache.load <- function(name) {
  path <- file.path("cache", paste0(name, ".rds"))
  if (file.exists(path)) readRDS(path) else NULL
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – German Credit
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1a. Data & model setup ───────────────────────────────────────────────────

#' Build the German Credit setup object.
#' Returns a list with:
#'   $data       – full german data frame
#'   $ciu        – CIU closure (with Traditional Banking vocabulary by default)
#'   $ciu.plain  – CIU closure without vocabulary (raw features)
#'   $pms        – list of partner models (Traditional Banking, Risk Assessment)
#'   $out.name   – "good"  (the "good credit" output)
#'   $target     – "credit_risk"
#'   $top.concepts – character vector of Traditional Banking ICs
#'   $top.concepts.ra – character vector of Risk Assessment ICs
#'   $focal.instances – named list of "interesting" instances for figures
german.setup <- function(use.cache = TRUE) {
  if (use.cache) {
    cached <- cache.load("german_setup")
    if (!is.null(cached)) return(cached)
  }

  data("german", package = "rchallenge")
  target <- "credit_risk"

  # Partner models (Traditional Banking + Risk Assessment)
  pms <- create.GermanCredit.partner_models(german)

  # Vocabulary for CIU (both hierarchies merged)
  voc <- list()
  for (pm in pms)
    voc <- c(voc, ciu.voc.from.graph(pm$graph,
                                     colnames(german)[colnames(german) != target]))

  # Train Random Forest via caret
  set.seed(42)
  kfoldcv <- trainControl(method = "cv", number = 10)
  rf.model <- train(credit_risk ~ ., german, method = "rf", trControl = kfoldcv)

  # CIU objects
  ciu.obj <- ciu.new(rf.model, credit_risk ~ ., german, vocabulary = voc)
  ciu.plain <- ciu.new(rf.model, credit_risk ~ ., german)

  # Interesting instances:
  #   i_good  – an instance predicted as "good" credit
  #   i_bad   – an instance predicted as "bad" credit
  #   i_border – borderline (close to 0.5 predicted probability)
  preds <- predict(rf.model, german, type = "prob")
  good.pred <- preds[, "good"]
  i_good   <- which(german$credit_risk == "good" & good.pred > 0.85)[1]
  i_bad    <- which(german$credit_risk == "bad"  & good.pred < 0.25)[1]
  i_border <- which(abs(good.pred - 0.5) < 0.05)[1]

  focal.instances <- list(
    good   = german[i_good,   colnames(german) != target, drop = FALSE],
    bad    = german[i_bad,    colnames(german) != target, drop = FALSE],
    border = german[i_border, colnames(german) != target, drop = FALSE]
  )

  su <- list(
    data             = german,
    ciu              = ciu.obj,
    ciu.plain        = ciu.plain,
    pms              = pms,
    out.name         = "good",
    target           = target,
    top.concepts     = ciu.kg.get.child.features(pms[["Traditional Banking"]]$graph,
                                                 "Credit Worthiness"),
    top.concepts.ra  = ciu.kg.get.child.features(pms[["Risk Assessment"]]$graph,
                                                 "Credit Risk"),
    focal.instances  = focal.instances
  )
  cache.save(su, "german_setup")
  su
}

# Shorthand: get an instance from the su list by name or index
get.instance <- function(su, i) {
  if (is.character(i)) return(su$focal.instances[[i]])
  su$data[i, colnames(su$data) != su$target, drop = FALSE]
}

# ── 1b. Figure helpers ────────────────────────────────────────────────────────

# Common theme tweaks for publication
pub.theme <- function() {
  theme_bw() +
    theme(text = element_text(size = 9),
          strip.text = element_text(size = 8),
          plot.title = element_text(size = 9, face = "bold"))
}

# ── 1c. Five-question figures (German Credit) ─────────────────────────────────

#' Q1 "Why?" – PI plot (Potential Influence): CI bar + CU fill
fig.why.pi <- function(su, i = "good", name = "fig_why_pi",
                       concepts = su$top.concepts,
                       n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             plot.mode = "overlap",
                             sort = "CI") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q1b "Why?" – influence-only barplot
fig.why.influence <- function(su, i = "good", name = "fig_why_influence",
                              concepts = su$top.concepts,
                              neutral.CU = 0.5, n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Key comparison figure: PI (left) vs. influence (right) with ALIGNED rows.
#' Uses native row.order parameter (no force.row.order workaround needed).
fig.pi.vs.influence <- function(su, i = "good", name = "fig_pi_vs_influence",
                                concepts = su$top.concepts,
                                neutral.CU = 0.5, n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)

  # Build PI plot and extract CI-sorted row order
  p.pi <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                output.names = su$out.name,
                                plot.mode = "overlap",
                                sort = "CI")

  # Extract the factor levels from the PI plot (bottom-to-top order)
  ci.cu.data <- p.pi$data
  ord <- levels(ci.cu.data$feature.labels)  # already sorted by CI ascending

  # Rebuild both panels with explicit row.order so they share the same axis
  p.pi  <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 plot.mode = "overlap",
                                 row.order = ord) +
    ggtitle("a  Importance & Utility (PI)") + pub.theme()

  p.inf <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 use.influence = TRUE,
                                 neutral.CU = neutral.CU,
                                 row.order = ord) +
    ggtitle("b  Contextual influence") + pub.theme()

  combined <- p.pi + p.inf
  save.fig(combined, name)
  combined
}

#' Q2 "Why not?" – influence for a BAD instance
fig.why.not <- function(su, i = "bad", name = "fig_why_not",
                        concepts = su$top.concepts,
                        neutral.CU = 0.5, n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") +
    ggtitle("Why not? (bad credit instance)") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q3 "What if?" – IO (Input-Output) plot for one feature
#' Uses ggplot.ciu with ind.input (singular!)
fig.whatif.io <- function(su, i = "good", ind.input = 5, name = "fig_whatif_io",
                          n.samples = 500) {
  instance <- get.instance(su, i)
  p <- su$ciu.plain$ggplot.ciu(instance, ind.input = ind.input,
                               illustrate.CIU = TRUE,
                               n.points = 40) + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q5 "Why A not B?" – contrastive explanation between two instances.
#'
#' ciu.contrastive() computes phi_A(CU_B) — the influence of each feature in
#' instance A using instance B's CU as the neutral reference. Features where A
#' has higher CU than B get positive contrastive influence; the result answers
#' "what makes A better/different from B?"
#'
#' out.ind: index of the output to explain (default 2 = "good" for German Credit
#' because caret RF orders factor levels alphabetically: bad=1, good=2).
#' For a single-output regression (Ames) use out.ind=1.
fig.why.a.not.b <- function(su, i.a = "good", i.b = "bad",
                            name = "fig_why_a_not_b",
                            concepts = su$top.concepts,
                            out.ind = 2,
                            n.samples = 500) {
  inst.a <- get.instance(su, i.a)
  inst.b <- get.instance(su, i.b)

  meta.a <- su$ciu$meta.explain(inst.a, concepts.to.explain = concepts,
                                n.samples = n.samples)
  meta.b <- su$ciu$meta.explain(inst.b, concepts.to.explain = concepts,
                                n.samples = n.samples)

  # Extract the relevant output row from each meta result
  frame.a <- ciu.list.to.frame(meta.a$ciuvals, out.ind = out.ind)
  frame.b <- ciu.list.to.frame(meta.b$ciuvals, out.ind = out.ind)

  # Contrastive: phi_A with CU_B as neutral — positive means A is better here
  contr <- ciu.contrastive(frame.a, frame.b)

  # Label instances by their row name or the focal-instance key
  lbl.a <- if (is.character(i.a)) i.a else rownames(inst.a)
  lbl.b <- if (is.character(i.b)) i.b else rownames(inst.b)

  p <- ciu.ggplot.contrastive(meta.a, contr,
                              instance.names = c(lbl.a, lbl.b),
                              question = "Why?") + pub.theme()
  save.fig(p, name)
  p
}

# ── 1d. Vocabulary comparison (Traditional Banking vs. Risk Assessment) ────────

#' Show the same instance explained with two different IC vocabularies
#' side by side, demonstrating vocabulary transparency.
fig.vocabulary.comparison <- function(su, i = "good",
                                      name = "fig_vocabulary_comparison",
                                      n.samples = 500) {
  instance <- get.instance(su, i)

  # Traditional Banking vocabulary
  meta.tb <- su$ciu$meta.explain(instance,
                                 concepts.to.explain = su$top.concepts,
                                 n.samples = n.samples)
  p.tb <- su$ciu$ggplot.col.ciu(ciu.meta = meta.tb,
                                output.names = su$out.name,
                                plot.mode = "overlap",
                                sort = "CI") +
    ggtitle("a  Traditional Banking") + pub.theme()

  # Risk Assessment vocabulary
  meta.ra <- su$ciu$meta.explain(instance,
                                 concepts.to.explain = su$top.concepts.ra,
                                 n.samples = n.samples)
  p.ra <- su$ciu$ggplot.col.ciu(ciu.meta = meta.ra,
                                output.names = su$out.name,
                                plot.mode = "overlap",
                                sort = "CI") +
    ggtitle("b  Risk Assessment") + pub.theme()

  combined <- p.tb + p.ra
  save.fig(combined, name)
  combined
}

# ── 1e. CI vs |influence| scatter (German Credit) ────────────────────────────

#' Scan all (or a random sample of) instances and collect CI and |phi| for
#' each feature × instance combination.
#' Returns a data frame with columns: instance, feature, CI, phi, absphi, CU.
scan.ci.vs.influence <- function(su, n.inst = 100, seed = 42,
                                 concepts = su$top.concepts,
                                 neutral.CU = 0.5,
                                 n.samples = 200,
                                 use.cache = TRUE,
                                 cache.name = "german_scan") {
  if (use.cache) {
    cached <- cache.load(cache.name)
    if (!is.null(cached)) return(cached)
  }

  set.seed(seed)
  n.total <- nrow(su$data)
  idx <- sample(n.total, min(n.inst, n.total))

  rows <- lapply(idx, function(ii) {
    inst <- su$data[ii, colnames(su$data) != su$target, drop = FALSE]
    meta <- tryCatch(
      su$ciu$meta.explain(inst, concepts.to.explain = concepts,
                          n.samples = n.samples),
      error = function(e) NULL
    )
    if (is.null(meta)) return(NULL)
    do.call(rbind, lapply(seq_along(concepts), function(j) {
      res <- meta$ciuvals[[j]]
      # Filter to the requested output (NULL = use first/only output row)
      if (!is.null(su$out.name))
        res <- res[rownames(res) == su$out.name, , drop = FALSE]
      else
        res <- res[1, , drop = FALSE]
      if (nrow(res) == 0) return(NULL)
      phi <- ciu.contextual.influence(res, neutral.CU = neutral.CU)
      data.frame(
        instance  = ii,
        feature   = concepts[j],
        CI        = res$CI,
        CU        = res$CU,
        phi       = phi,
        absphi    = abs(phi),
        divergence = res$CI - 2 * abs(phi)  # CI × (1 - |2CU-1|)
      )
    }))
  })
  df <- do.call(rbind, Filter(Negate(is.null), rows))
  cache.save(df, cache.name)
  df
}

#' Proportion of instances where CI ranking ≠ |phi| ranking, per feature.
ranking.disagreement <- function(scan.df) {
  # For each instance, rank features by CI and by |phi| separately
  per.inst <- split(scan.df, scan.df$instance)
  disagreement <- vapply(per.inst, function(d) {
    r.ci  <- rank(-d$CI)
    r.phi <- rank(-d$absphi)
    mean(r.ci != r.phi)
  }, numeric(1))
  data.frame(instance = names(disagreement),
             disagreement = disagreement)
}

#' Scatter: CI (x) vs |phi| (y) coloured by CU.
#' Points on the diagonal → influence fully captures importance.
#' Points away from diagonal → divergence (importance invisible to SHAP).
fig.ci.vs.influence <- function(scan.df, name = "fig_ci_vs_influence",
                                dataset.label = "German Credit") {
  p <- ggplot(scan.df, aes(x = CI, y = absphi, colour = CU)) +
    geom_point(alpha = 0.4, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
    scale_colour_gradient2(low = "red", mid = "yellow", high = "darkgreen",
                           midpoint = 0.5, limits = c(0, 1)) +
    labs(x = "Contextual Importance (CI)",
         y = "Contextual |influence| (|φ|)",
         colour = "CU",
         title = paste0(dataset.label,
                        ": CI vs |influence| across instances & features"),
         subtitle = "Points below diagonal: importance invisible to influence methods") +
    theme_bw() +
    theme(text = element_text(size = 9))
  save.fig(p, name)
  p
}

#' Divergence distribution: CI − 2|phi| histogram.
#' A right-skewed distribution means most importance is NOT captured by influence.
fig.divergence.hist <- function(scan.df, name = "fig_divergence_hist",
                                dataset.label = "German Credit") {
  p <- ggplot(scan.df, aes(x = divergence)) +
    geom_histogram(bins = 40, fill = "steelblue", colour = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "firebrick") +
    labs(x = "Divergence  CI − 2|φ|",
         y = "Count",
         title = paste0(dataset.label, ": divergence distribution"),
         subtitle = "Positive values: importance not visible as influence") +
    theme_bw() +
    theme(text = element_text(size = 9))
  save.fig(p, name, width = FIG_W_1COL)
  p
}

# ── 1f. Master function: all German Credit figures ────────────────────────────
make.all.german.figures <- function(su = NULL, n.scan = 200, n.samples = 500) {
  if (is.null(su)) su <- german.setup()

  message("German Credit: Q1 PI plot ...")
  fig.why.pi(su, n.samples = n.samples)

  message("German Credit: Q1 influence plot ...")
  fig.why.influence(su, n.samples = n.samples)

  message("German Credit: PI vs influence (aligned) ...")
  fig.pi.vs.influence(su, n.samples = n.samples)

  message("German Credit: Q2 Why not? ...")
  fig.why.not(su, n.samples = n.samples)

  message("German Credit: Q3 What if? IO plot ...")
  fig.whatif.io(su)

  message("German Credit: Q5 Why A not B? ...")
  fig.why.a.not.b(su, n.samples = n.samples)

  message("German Credit: Vocabulary comparison ...")
  fig.vocabulary.comparison(su, n.samples = n.samples)

  message("German Credit: CI vs |influence| scan (n=", n.scan, ") ...")
  scan.df <- scan.ci.vs.influence(su, n.inst = n.scan, n.samples = 150)
  fig.ci.vs.influence(scan.df, dataset.label = "German Credit")
  fig.divergence.hist(scan.df, dataset.label = "German Credit")

  message("German Credit figures done.")
  invisible(su)
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Ames Housing
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. Data & model setup ───────────────────────────────────────────────────

#' Build the Ames Housing setup object.
#' Returns a list with fields analogous to german.setup().
ames.setup <- function(use.cache = TRUE) {
  if (use.cache) {
    cached <- cache.load("ames_setup")
    if (!is.null(cached)) return(cached)
  }

  ames_data <- data.frame(make_ames())
  target <- "Sale_Price"
  ames_formula <- as.formula(paste(target, "~."))

  # Partner models (Default KG from TestAmesKnowledgeGraph.R)
  pms <- create.Ames.KGs(ames_data)

  # Train GBM via caret (10-fold CV)
  set.seed(42)
  kfoldcv <- trainControl(method = "cv", number = 10)
  gbm.model <- train(ames_formula, ames_data, method = "gbm",
                     trControl = kfoldcv, verbose = FALSE)

  # CIU with Default vocabulary
  voc <- ciu.voc.from.graph(pms[["Default"]]$graph,
                            colnames(ames_data)[colnames(ames_data) != target])
  ciu.obj   <- ciu.new(gbm.model, ames_formula, ames_data, vocabulary = voc)
  ciu.plain <- ciu.new(gbm.model, ames_formula, ames_data)

  # Default vocabulary top-level ICs
  top.concepts.default <- ciu.kg.get.child.features(pms[["Default"]]$graph,
                                                    "SalesPrice_def_voc")

  # Choose focal instances
  preds <- predict(gbm.model, ames_data)
  actual <- ames_data[[target]]
  resid  <- actual - preds

  # High-value house well-predicted
  i_high   <- which(preds > quantile(preds, 0.90) & abs(resid / actual) < 0.05)[1]
  # Low-value house well-predicted
  i_low    <- which(preds < quantile(preds, 0.10) & abs(resid / actual) < 0.05)[1]
  # Median house
  i_median <- which(abs(preds - median(preds)) < 5000)[1]

  focal.instances <- list(
    high   = ames_data[i_high,   colnames(ames_data) != target, drop = FALSE],
    low    = ames_data[i_low,    colnames(ames_data) != target, drop = FALSE],
    median = ames_data[i_median, colnames(ames_data) != target, drop = FALSE]
  )

  su <- list(
    data             = ames_data,
    ciu              = ciu.obj,
    ciu.plain        = ciu.plain,
    pms              = pms,
    out.name         = NULL,   # NULL = include all outputs (regression has no named class)
    target           = target,
    top.concepts     = top.concepts.default,
    focal.instances  = focal.instances
  )
  cache.save(su, "ames_setup")
  su
}

# ── 2b. Five-question figures (Ames Housing) ──────────────────────────────────

fig.ames.why.pi <- function(su, i = "high", name = "fig_ames_why_pi",
                            n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             plot.mode = "overlap",
                             sort = "CI") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

fig.ames.why.influence <- function(su, i = "high", name = "fig_ames_why_influence",
                                   neutral.CU = 0.5, n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

fig.ames.pi.vs.influence <- function(su, i = "high",
                                     name = "fig_ames_pi_vs_influence",
                                     neutral.CU = 0.5, n.samples = 500) {
  instance <- get.instance(su, i)
  meta <- su$ciu$meta.explain(instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)

  # Get CI-sorted order from initial PI plot
  p.pi0 <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 plot.mode = "overlap",
                                 sort = "CI")
  ord <- levels(p.pi0$data$feature.labels)

  p.pi  <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 plot.mode = "overlap",
                                 row.order = ord) +
    ggtitle("a  Importance & Utility (PI)") + pub.theme()
  p.inf <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 use.influence = TRUE,
                                 neutral.CU = neutral.CU,
                                 row.order = ord) +
    ggtitle("b  Contextual influence") + pub.theme()

  combined <- p.pi + p.inf
  save.fig(combined, name)
  combined
}

#' Q3 "What if?" IO plot for a numeric Ames feature (default: Gr_Liv_Area)
fig.ames.whatif.io <- function(su, i = "median",
                               ind.input = which(colnames(su$data) == "Gr_Liv_Area") - 0L,
                               name = "fig_ames_whatif_io") {
  # Note: ind.input must be the column index in the DATA (excl. target here)
  # ciu.plain was built with Sale_Price ~ . so column indexing matches ames_data
  # but the closure sees all columns including target; find correct index:
  all.cols <- colnames(su$data)
  feat.name <- colnames(su$data[, colnames(su$data) != su$target])[ind.input]
  ind.in.full <- which(all.cols == feat.name)

  instance <- get.instance(su, i)
  # ggplot.ciu needs all columns (incl. target) for internal min/max lookup.
  # get.instance strips the target, so re-fetch the full row by row index.
  row.idx <- if (is.character(i)) {
    which(rownames(su$data) == rownames(instance))[1]
  } else i
  full.inst <- su$data[row.idx, ]
  p <- su$ciu.plain$ggplot.ciu(full.inst, ind.input = ind.in.full,
                               illustrate.CIU = TRUE,
                               n.points = 40) + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q5 "Why A not B?" – two houses, contrastive at IC level.
#' Ames is single-output regression so out.ind=1 always.
fig.ames.why.a.not.b <- function(su, i.a = "high", i.b = "low",
                                 name = "fig_ames_why_a_not_b",
                                 out.ind = 1,
                                 n.samples = 500) {
  inst.a <- get.instance(su, i.a)
  inst.b <- get.instance(su, i.b)

  meta.a <- su$ciu$meta.explain(inst.a, concepts.to.explain = su$top.concepts,
                                n.samples = n.samples)
  meta.b <- su$ciu$meta.explain(inst.b, concepts.to.explain = su$top.concepts,
                                n.samples = n.samples)

  frame.a <- ciu.list.to.frame(meta.a$ciuvals, out.ind = out.ind)
  frame.b <- ciu.list.to.frame(meta.b$ciuvals, out.ind = out.ind)
  contr <- ciu.contrastive(frame.a, frame.b)

  p <- ciu.ggplot.contrastive(meta.a, contr,
                              instance.names = c(i.a, i.b),
                              question = "Why?") + pub.theme()
  save.fig(p, name)
  p
}

# ── 2c. CI vs |influence| scatter (Ames Housing) ─────────────────────────────

#' Ames-specific CI vs |phi| scan (uses top-level ICs by default).
#' With 80 features at raw level the plot is very dense; ICs are more readable.
scan.ames.ci.vs.influence <- function(su, n.inst = 150, seed = 42,
                                      concepts = su$top.concepts,
                                      neutral.CU = 0.5,
                                      n.samples = 150,
                                      use.cache = TRUE,
                                      cache.name = "ames_scan") {
  # Reuse the generic German Credit scanner – it only depends on su$ciu,
  # su$data, su$target, su$out.name.
  scan.ci.vs.influence(su, n.inst = n.inst, seed = seed,
                       concepts = concepts,
                       neutral.CU = neutral.CU,
                       n.samples = n.samples,
                       use.cache = use.cache,
                       cache.name = cache.name)
}

# ── 2d. Master function: all Ames figures ─────────────────────────────────────
make.all.ames.figures <- function(su = NULL, n.scan = 200, n.samples = 500) {
  if (is.null(su)) su <- ames.setup()

  message("Ames: Q1 PI plot ...")
  fig.ames.why.pi(su, n.samples = n.samples)

  message("Ames: Q1 influence plot ...")
  fig.ames.why.influence(su, n.samples = n.samples)

  message("Ames: PI vs influence (aligned) ...")
  fig.ames.pi.vs.influence(su, n.samples = n.samples)

  message("Ames: Q3 What if? IO plot ...")
  fig.ames.whatif.io(su)

  message("Ames: Q5 Why A not B? ...")
  fig.ames.why.a.not.b(su, n.samples = n.samples)

  message("Ames: CI vs |influence| scan (n=", n.scan, ") ...")
  scan.df <- scan.ames.ci.vs.influence(su, n.inst = n.scan, n.samples = 150)
  fig.ci.vs.influence(scan.df,
                      name = "fig_ames_ci_vs_influence",
                      dataset.label = "Ames Housing")
  fig.divergence.hist(scan.df,
                      name = "fig_ames_divergence_hist",
                      dataset.label = "Ames Housing")

  message("Ames figures done.")
  invisible(su)
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Combined / cross-dataset figures
# ═══════════════════════════════════════════════════════════════════════════════

#' Overlay CI vs |phi| scatter for both datasets in one figure.
#' Useful for showing that the divergence is a universal phenomenon.
fig.combined.ci.vs.influence <- function(german.scan, ames.scan,
                                         name = "fig_combined_ci_vs_influence") {
  german.scan$dataset <- "German Credit"
  ames.scan$dataset   <- "Ames Housing"
  df <- rbind(german.scan, ames.scan)

  p <- ggplot(df, aes(x = CI, y = absphi, colour = CU)) +
    geom_point(alpha = 0.35, size = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
    scale_colour_gradient2(low = "red", mid = "yellow", high = "darkgreen",
                           midpoint = 0.5, limits = c(0, 1)) +
    facet_wrap(~dataset) +
    labs(x = "Contextual Importance (CI)",
         y = "Contextual |influence| (|φ|)",
         colour = "CU",
         title = "CI vs |influence|: divergence across datasets",
         subtitle = "Diagonal = influence equals importance; below = invisible importance") +
    theme_bw() +
    theme(text = element_text(size = 9), strip.text = element_text(size = 8))
  save.fig(p, name)
  p
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Run everything
# ═══════════════════════════════════════════════════════════════════════════════

#' Generate ALL paper figures.
#' Set use.cache=FALSE the first time (trains models), TRUE afterwards.
make.all.figures <- function(use.cache = TRUE) {
  message("=== German Credit ===")
  su.german <- german.setup(use.cache = use.cache)
  make.all.german.figures(su.german)

  message("=== Ames Housing ===")
  su.ames <- ames.setup(use.cache = use.cache)
  make.all.ames.figures(su.ames)

  message("=== Combined figures ===")
  g.scan <- cache.load("german_scan")
  a.scan <- cache.load("ames_scan")
  if (!is.null(g.scan) && !is.null(a.scan))
    fig.combined.ci.vs.influence(g.scan, a.scan)

  message("All figures done. See empirical/figures/")
}

# ── Notes on CIU package API (master branch) ──────────────────────────────────
# ciu.ggplot.col / ggplot.col.ciu:
#   Parameters: sort=NULL/"CI"/"CU"/"phi"/"absphi", decreasing=FALSE,
#               row.order=<character vector, bottom-to-top> (native, no workaround needed)
#   force.row.order() workaround from earlier versions is NOW OBSOLETE.
#
# ggplot.ciu (inside CIU closure):
#   Parameter is ind.input (SINGULAR, not ind.inputs)
#   illustrate.CIU=FALSE is correct (not illustrate.ciu)
#
# ciu.contrastive:
#   Takes two data frames from ciu.list.to.frame()
#
# ciu.ggplot.contrastive:
#   Takes ciu.meta.result, contrastive df, instance.names, question
#
# ciu.kg.get.child.features(graph, root.concept.name):
#   Returns character vector of direct children (ICs or features)
#
# ciu.list.to.frame(ciu.list, out.ind=1):
#   Extracts one output row from each ciuvals entry; result is feature × CIU columns
