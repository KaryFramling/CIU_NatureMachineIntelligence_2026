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
# ── Nature figure dimensions (inches) ────────────────────────────────────────
FIG_W_1COL <- 3.50
FIG_W_2COL <- 7.20
FIG_H      <- 4.20

# Save helper – cairo_pdf for production-quality vector output
save.fig <- function(p, name, width = FIG_W_2COL, height = FIG_H) {
  dir.create("figures", recursive = TRUE, showWarnings = FALSE)
  path <- file.path("figures", paste0(name, ".pdf"))
  cairo_pdf(path, width = width, height = height)
  print(p)
  dev.off()
  invisible(path)
}

# Cache helpers
cache.save <- function(obj, name) {
  dir.create("cache", recursive = TRUE, showWarnings = FALSE)
  saveRDS(obj, file.path("cache", paste0(name, ".rds")))
}
cache.load <- function(name) {
  path <- file.path("cache", paste0(name, ".rds"))
  if (!dir.exists("cache")) return(NULL)
  if (file.exists(path)) readRDS(path) else NULL
}

# ── CIU perturbation samples per feature / intermediate concept ─────────────
# 1000 handles intermediate concepts (several features varied jointly) well.
N_SAMPLES <- 1000
FIG_SEED  <- 42     # seed for the perturbation sampling in each figure

# ── PI-plot colours ───────────────────────────────────────────────────────────
# "overlap" mode: light-grey CI bar (potential range) with the CI×CU bar on top,
# coloured by CU on the red–yellow–green scale. cu.colours = NULL is what
# switches the CU bar from a fixed colour to the CU gradient.
PI.CI.COLOURS <- c("grey85", "grey50", "1")
PI.CU.COLOURS <- NULL

# ── Random Forest smoothing / calibration ────────────────────────────────────
# Raw RF class probabilities are vote fractions: step-like along each feature
# and, when evaluated on training data, pushed towards 0/1. Two remedies:
#   1. more trees + larger terminal nodes -> smoother response surface;
#   2. Platt scaling (logistic sigmoid fitted on the out-of-bag votes) ->
#      well-calibrated, "softened" probabilities.
RF_NTREE    <- 1000
RF_NODESIZE <- 10

#' Platt scaling fitted on out-of-bag RF votes.
#' Returns a function p -> calibrated p.
platt.calibrate <- function(rf.train, y, positive = "good") {
  fm    <- rf.train$finalModel
  eps   <- 1 / (2 * fm$ntree)
  clamp <- function(p) pmin(pmax(p, eps), 1 - eps)
  z     <- qlogis(clamp(fm$votes[, positive]))      # OOB vote fractions
  ok    <- is.finite(z)
  fit   <- glm(as.integer(y[ok] == positive) ~ z[ok], family = binomial)
  a <- unname(coef(fit)[1]); b <- unname(coef(fit)[2])
  message(sprintf("Platt scaling: a = %.3f, b = %.3f", a, b))
  function(p) plogis(a + b * qlogis(clamp(p)))
}

#' Pick the instance that best illustrates "importance is not influence".
#' Candidates: the n.cand instances whose predicted output is closest to the
#' average prediction. Among them, choose the one with the most concepts that
#' are important (CI > ci.min) yet have near-zero influence (|phi| < phi.zero);
#' ties broken by the total divergence sum(CI - 2|phi|).
select.typical.instance <- function(ciu.obj, data, target, preds,
                                    concepts, out.ind = 1,
                                    n.cand = 40, n.samples = N_SAMPLES,
                                    ci.min = 0.10, phi.zero = 0.02,
                                    neutral.CU = 0.5) {
  cand <- order(abs(preds - mean(preds)))[seq_len(min(n.cand, length(preds)))]
  scores <- t(vapply(cand, function(ii) {
    inst <- data[ii, colnames(data) != target, drop = FALSE]
    meta <- ciu.obj$meta.explain(inst, concepts.to.explain = concepts,
                                 n.samples = n.samples)
    fr  <- ciu.list.to.frame(meta$ciuvals, out.ind = out.ind)
    phi <- fr$CI * (fr$CU - neutral.CU)
    c(n.zero     = sum(fr$CI > ci.min & abs(phi) < phi.zero),
      divergence = sum(fr$CI - 2 * abs(phi)))
  }, numeric(2)))
  best <- order(-scores[, "n.zero"], -scores[, "divergence"])[1]
  message(sprintf("Typical instance: row %d, %d/%d concepts with CI>%.2f and |phi|<%.2f, divergence %.3f",
                  cand[best], scores[best, "n.zero"], length(concepts),
                  ci.min, phi.zero, scores[best, "divergence"]))
  cand[best]
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
  rf.model <- train(credit_risk ~ ., german, method = "rf", trControl = kfoldcv,
                    ntree = RF_NTREE, nodesize = RF_NODESIZE)

  # Softened (Platt-calibrated) probabilities used for ALL CIU computations
  calib    <- platt.calibrate(rf.model, german$credit_risk, positive = "good")
  pred.fun <- function(model, inputs) {
    q <- calib(predict(model, inputs, type = "prob")[, "good"])
    data.frame(bad = 1 - q, good = q)
  }

  # CIU objects
  ciu.obj   <- ciu.new(rf.model, credit_risk ~ ., german, vocabulary = voc,
                       predict.function = pred.fun)
  ciu.plain <- ciu.new(rf.model, credit_risk ~ ., german,
                       predict.function = pred.fun)

  top.concepts <- ciu.kg.get.child.features(pms[["Traditional Banking"]]$graph,
                                            "Credit Worthiness")

  # Interesting instances:
  #   good    – confidently "good" (top decile of predicted p_good)
  #   bad     – confidently "bad"  (bottom decile)
  #   border  – closest to p_good = 0.5
  #   typical – output close to the average prediction AND as many important
  #             concepts as possible with near-zero influence (CU ≈ 0.5);
  #             this is the instance where PI and influence differ most.
  good.pred <- pred.fun(rf.model, german)[, "good"]
  i_good   <- which(german$credit_risk == "good" &
                      good.pred >= quantile(good.pred, 0.90))[1]
  i_bad    <- which(german$credit_risk == "bad"  &
                      good.pred <= quantile(good.pred, 0.10))[1]
  i_border <- which.min(abs(good.pred - 0.5))
  i_typical <- select.typical.instance(ciu.obj, german, target, good.pred,
                                       concepts = top.concepts, out.ind = 2)

  focal.instances <- list(
    good    = german[i_good,    colnames(german) != target, drop = FALSE],
    bad     = german[i_bad,     colnames(german) != target, drop = FALSE],
    border  = german[i_border,  colnames(german) != target, drop = FALSE],
    typical = german[i_typical, colnames(german) != target, drop = FALSE]
  )
  message(sprintf("German Credit focal instances (row, p_good): good=%d (%.2f), bad=%d (%.2f), border=%d (%.2f), typical=%d (%.2f); mean p_good=%.2f",
                  i_good, good.pred[i_good], i_bad, good.pred[i_bad],
                  i_border, good.pred[i_border], i_typical, good.pred[i_typical],
                  mean(good.pred)))

  su <- list(
    data             = german,
    model            = rf.model,
    predict          = pred.fun,
    pred.out         = good.pred,
    ciu              = ciu.obj,
    ciu.plain        = ciu.plain,
    pms              = pms,
    out.name         = "good",
    target           = target,
    top.concepts     = top.concepts,
    top.concepts.ra  = ciu.kg.get.child.features(pms[["Risk Assessment"]]$graph,
                                                 "Credit Risk"),
    focal.instances  = focal.instances
  )
  cache.save(su, "german_setup")
  su
}

# CIU values are computed once per (dataset, instance, concepts, n.samples) and
# reused by every figure, so the same instance always shows identical values.
.meta.memo <- new.env()
cached.meta <- function(su, instance, concepts.to.explain, n.samples) {
  key <- paste(su$target, rownames(instance)[1],
               paste(concepts.to.explain, collapse = "|"), n.samples, sep = "#")
  if (!exists(key, envir = .meta.memo, inherits = FALSE)) {
    set.seed(FIG_SEED)
    assign(key, su$ciu$meta.explain(instance,
                                    concepts.to.explain = concepts.to.explain,
                                    n.samples = n.samples),
           envir = .meta.memo)
  }
  get(key, envir = .meta.memo, inherits = FALSE)
}

# Shorthand: get an instance from the su list by name or index
get.instance <- function(su, i) {
  if (is.character(i)) return(su$focal.instances[[i]])
  su$data[i, colnames(su$data) != su$target, drop = FALSE]
}

# ── 1b. Figure helpers ────────────────────────────────────────────────────────

# Common theme tweaks for publication
pub.theme <- function(legend = TRUE) {
  theme_bw() +
    theme(text = element_text(size = 9),
          strip.text = element_text(size = 8),
          plot.title = element_text(size = 9, face = "bold"),
          legend.position = if (legend) "right" else "none")
}

# ── 1c. Five-question figures (German Credit) ─────────────────────────────────

#' Q1 "Why?" – PI plot (Potential Influence): CI bar + CU fill
fig.why.pi <- function(su, i = "typical", name = "fig_why_pi",
                       concepts = su$top.concepts,
                       n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             show.input.values = FALSE,
                             plot.mode = "overlap",
                             ci.colours = PI.CI.COLOURS,
                             cu.colours = PI.CU.COLOURS,
                             sort = "CI") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q1b "Why?" – influence-only barplot
fig.why.influence <- function(su, i = "typical", name = "fig_why_influence",
                              concepts = su$top.concepts,
                              neutral.CU = 0.5, n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             show.input.values = FALSE,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") + pub.theme(legend = FALSE)
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Key comparison figure: PI (left) vs. influence (right) with ALIGNED rows.
#' Uses native row.order parameter (no force.row.order workaround needed).
fig.pi.vs.influence <- function(su, i = "typical", name = "fig_pi_vs_influence",
                                concepts = su$top.concepts,
                                neutral.CU = 0.5, n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)

  # Build PI plot and extract CI-sorted row order
  p.pi <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                output.names = su$out.name,
                                show.input.values = FALSE,
                                plot.mode = "overlap",
                                ci.colours = PI.CI.COLOURS,
                                cu.colours = PI.CU.COLOURS,
                                sort = "CI")

  # Extract the factor levels from the PI plot (bottom-to-top order)
  ci.cu.data <- p.pi$data
  ord <- levels(ci.cu.data$feature.labels)  # already sorted by CI ascending

  # Rebuild both panels with explicit row.order so they share the same axis
  p.pi  <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 show.input.values = FALSE,
                                 plot.mode = "overlap",
                                 ci.colours = PI.CI.COLOURS,
                                 cu.colours = PI.CU.COLOURS,
                                 row.order = ord) +
    ggtitle("Importance & Utility (PI)") + pub.theme()

  p.inf <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 show.input.values = FALSE,
                                 use.influence = TRUE,
                                 neutral.CU = neutral.CU,
                                 row.order = ord) +
    ggtitle("Contextual influence") + pub.theme(legend = FALSE)

  combined <- p.pi + p.inf
  save.fig(combined, name)
  combined
}

#' Q2 "Why not?" – influence for a BAD instance
fig.why.not <- function(su, i = "bad", name = "fig_why_not",
                        concepts = su$top.concepts,
                        neutral.CU = 0.5, n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             show.input.values = FALSE,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") +
    ggtitle("Why not? (bad credit instance)") + pub.theme(legend = FALSE)
  save.fig(p, name, width = FIG_W_1COL)
  p
}

#' Q3 "What if?" – IO (Input-Output) plot for one feature
#' Uses ggplot.ciu with ind.input (singular!)
fig.whatif.io <- function(su, i = "good", ind.input = 5, name = "fig_whatif_io",
                          n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
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
                            n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  inst.a <- get.instance(su, i.a)
  inst.b <- get.instance(su, i.b)

  meta.a <- cached.meta(su, inst.a, concepts.to.explain = concepts,
                                n.samples = n.samples)
  meta.b <- cached.meta(su, inst.b, concepts.to.explain = concepts,
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
                              question = "Why?") + pub.theme(legend = FALSE)
  save.fig(p, name, width = FIG_W_1COL, height = 3.0)
  p
}

# ── 1d. Vocabulary comparison (Traditional Banking vs. Risk Assessment) ────────

#' Show the same instance explained with two different IC vocabularies
#' side by side, demonstrating vocabulary transparency.
fig.vocabulary.comparison <- function(su, i = "typical",
                                      name = "fig_vocabulary_comparison",
                                      n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)

  # Traditional Banking vocabulary
  meta.tb <- cached.meta(su, instance,
                                 concepts.to.explain = su$top.concepts,
                                 n.samples = n.samples)
  p.tb <- su$ciu$ggplot.col.ciu(ciu.meta = meta.tb,
                                output.names = su$out.name,
                                show.input.values = FALSE,
                                plot.mode = "overlap",
                                ci.colours = PI.CI.COLOURS,
                                cu.colours = PI.CU.COLOURS,
                                sort = "CI") +
    ggtitle("Traditional Banking") + pub.theme()

  # Risk Assessment vocabulary
  meta.ra <- cached.meta(su, instance,
                                 concepts.to.explain = su$top.concepts.ra,
                                 n.samples = n.samples)
  p.ra <- su$ciu$ggplot.col.ciu(ciu.meta = meta.ra,
                                output.names = su$out.name,
                                show.input.values = FALSE,
                                plot.mode = "overlap",
                                ci.colours = PI.CI.COLOURS,
                                cu.colours = PI.CU.COLOURS,
                                sort = "CI") +
    ggtitle("Risk Assessment") + pub.theme()

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
                                 n.samples = N_SAMPLES,
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
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  p <- ggplot(scan.df, aes(x = CI, y = absphi, colour = CU)) +
    geom_point(alpha = 0.4, size = 0.8) +
    geom_abline(slope = 0.5, intercept = 0, linetype = "dashed", colour = "grey40") +
    scale_colour_gradient2(low = "red", mid = "yellow", high = "darkgreen",
                           midpoint = 0.5, limits = c(0, 1)) +
    labs(x = "Contextual Importance (CI)",
         y = "Contextual |influence| (|φ|)",
         colour = "CU") +   # envelope |phi| = CI/2 (dashed); title in figure legend
    theme_bw() +
    theme(text = element_text(size = 9))
  save.fig(p, name, width = FIG_W_1COL, height = 3.0)
  p
}

#' Divergence distribution: CI − 2|phi| histogram.
#' A right-skewed distribution means most importance is NOT captured by influence.
fig.divergence.hist <- function(scan.df, name = "fig_divergence_hist",
                                dataset.label = "German Credit") {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  p <- ggplot(scan.df, aes(x = divergence)) +
    geom_histogram(bins = 40, fill = "steelblue", colour = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "firebrick") +
    labs(x = "Divergence  CI − 2|φ|",
         y = "Count",
         title = dataset.label) +
    theme_bw() +
    theme(text = element_text(size = 9))
  save.fig(p, name, width = FIG_W_1COL)
  p
}

# ── 1f. Master function: all German Credit figures ────────────────────────────
make.all.german.figures <- function(su = NULL, n.scan = 200, n.samples = N_SAMPLES,
                                    use.cache = TRUE) {
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
  scan.df <- scan.ci.vs.influence(su, n.inst = n.scan, n.samples = N_SAMPLES,
                                  use.cache = use.cache)
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
                            n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             show.input.values = FALSE,
                             plot.mode = "overlap",
                             ci.colours = PI.CI.COLOURS,
                             cu.colours = PI.CU.COLOURS,
                             sort = "CI") + pub.theme()
  save.fig(p, name, width = FIG_W_1COL)
  p
}

fig.ames.why.influence <- function(su, i = "high", name = "fig_ames_why_influence",
                                   neutral.CU = 0.5, n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)
  p <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                             output.names = su$out.name,
                             show.input.values = FALSE,
                             use.influence = TRUE,
                             neutral.CU = neutral.CU,
                             sort = "absphi") + pub.theme(legend = FALSE)
  save.fig(p, name, width = FIG_W_1COL)
  p
}

fig.ames.pi.vs.influence <- function(su, i = "high",
                                     name = "fig_ames_pi_vs_influence",
                                     neutral.CU = 0.5, n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  instance <- get.instance(su, i)
  meta <- cached.meta(su, instance,
                              concepts.to.explain = su$top.concepts,
                              n.samples = n.samples)

  # Get CI-sorted order from initial PI plot
  p.pi0 <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 show.input.values = FALSE,
                                 plot.mode = "overlap",
                                 ci.colours = PI.CI.COLOURS,
                                 cu.colours = PI.CU.COLOURS,
                                 sort = "CI")
  ord <- levels(p.pi0$data$feature.labels)

  p.pi  <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 show.input.values = FALSE,
                                 plot.mode = "overlap",
                                 ci.colours = PI.CI.COLOURS,
                                 cu.colours = PI.CU.COLOURS,
                                 row.order = ord) +
    ggtitle("Importance & Utility (PI)") + pub.theme()
  p.inf <- su$ciu$ggplot.col.ciu(ciu.meta = meta,
                                 output.names = su$out.name,
                                 show.input.values = FALSE,
                                 use.influence = TRUE,
                                 neutral.CU = neutral.CU,
                                 row.order = ord) +
    ggtitle("Contextual influence") + pub.theme(legend = FALSE)

  combined <- p.pi + p.inf
  save.fig(combined, name)
  combined
}

#' Q3 "What if?" IO plot for a numeric Ames feature (default: Gr_Liv_Area)
fig.ames.whatif.io <- function(su, i = "median",
                               ind.input = which(colnames(su$data) == "Gr_Liv_Area") - 0L,
                               name = "fig_ames_whatif_io") {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
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
                                 n.samples = N_SAMPLES) {
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  inst.a <- get.instance(su, i.a)
  inst.b <- get.instance(su, i.b)

  meta.a <- cached.meta(su, inst.a, concepts.to.explain = su$top.concepts,
                                n.samples = n.samples)
  meta.b <- cached.meta(su, inst.b, concepts.to.explain = su$top.concepts,
                                n.samples = n.samples)

  frame.a <- ciu.list.to.frame(meta.a$ciuvals, out.ind = out.ind)
  frame.b <- ciu.list.to.frame(meta.b$ciuvals, out.ind = out.ind)
  contr <- ciu.contrastive(frame.a, frame.b)

  p <- ciu.ggplot.contrastive(meta.a, contr,
                              instance.names = c(i.a, i.b),
                              question = "Why?") + pub.theme(legend = FALSE)
  save.fig(p, name, width = FIG_W_1COL, height = 3.0)
  p
}

# ── 2c. CI vs |influence| scatter (Ames Housing) ─────────────────────────────

#' Ames-specific CI vs |phi| scan (uses top-level ICs by default).
#' With 80 features at raw level the plot is very dense; ICs are more readable.
scan.ames.ci.vs.influence <- function(su, n.inst = 150, seed = 42,
                                      concepts = su$top.concepts,
                                      neutral.CU = 0.5,
                                      n.samples = N_SAMPLES,
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
make.all.ames.figures <- function(su = NULL, n.scan = 200, n.samples = N_SAMPLES,
                                  use.cache = TRUE) {
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
  scan.df <- scan.ames.ci.vs.influence(su, n.inst = n.scan, n.samples = N_SAMPLES,
                                       use.cache = use.cache)
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
  set.seed(FIG_SEED)   # identical CIU estimates on every run and across figures
  german.scan$dataset <- "German Credit"
  ames.scan$dataset   <- "Ames Housing"
  df <- rbind(german.scan, ames.scan)

  p <- ggplot(df, aes(x = CI, y = absphi, colour = CU)) +
    geom_point(alpha = 0.35, size = 0.7) +
    geom_abline(slope = 0.5, intercept = 0, linetype = "dashed", colour = "grey40") +
    scale_colour_gradient2(low = "red", mid = "yellow", high = "darkgreen",
                           midpoint = 0.5, limits = c(0, 1)) +
    facet_wrap(~dataset) +
    labs(x = "Contextual Importance (CI)",
         y = "Contextual |influence| (|φ|)",
         colour = "CU") +   # envelope |phi| = CI/2 (dashed)
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
  make.all.german.figures(su.german, use.cache = use.cache)

  message("=== Ames Housing ===")
  su.ames <- ames.setup(use.cache = use.cache)
  make.all.ames.figures(su.ames, use.cache = use.cache)

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
