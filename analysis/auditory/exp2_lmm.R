# Experiment 2: mixed models of the grammar test days (lme4)
#
# Called by exp2_lmm.py with the table of arm-blocks (one row per arm, sound
# block and session). Each day is fitted on its own, over the shuffled blocks
# 2 to 4 (primary: the first block's layout was fixed, EE in arms 1 to 3) and
# over all four blocks; the arm is a random effect throughout, which absorbs the
# preference for places.
#
#   staying:  log mean visit length ~ environment * level + block, weighted by
#             visits, grammar arms; a contrast is a ratio of visit lengths
#   entering: Poisson GLMM of entries per arm-block, with an observation-level
#             random effect; a contrast is a ratio of entry rates
#   grammar:  grammar A against B in the same model as the environment (the
#             counterbalancing crosses them), a preference for one grammar's
#             sound whatever the cage it went with
#   class:    silent arm, grammar arms, vocalisation, log mean visit length
#
# Random effects start from the fullest structure and are simplified while the
# fit is singular or fails to converge. Intervals are unadjusted 95%; p values
# across the three levels are Holm-corrected.
#
#   Rscript exp2_lmm.R <arm_blocks.csv> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
emm_options(lmer.df = "satterthwaite")

d0 <- read.csv(args[1], stringsAsFactors = FALSE)
d0$tier <- factor(d0$tier, levels = c("dominant", "secondary", "rare"))
d0$environment <- factor(d0$environment, levels = c("SC", "EE"))
d0$group <- factor(d0$group)
d0$grammar <- factor(d0$grammar, levels = c("B", "A"))
d0$arm <- factor(d0$arm)
sink(file.path(args[2], "lmm_results.txt"), split = TRUE)
cat("Experiment 2, mixed models; lme4", as.character(packageVersion("lme4")), "\n")

ok <- function(m) !isSingular(m) && is.null(m@optinfo$conv$lme4$messages)
lctl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
gctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

fit_lmer <- function(fixed, randoms, data) {
  for (r in randoms) {
    m <- suppressMessages(suppressWarnings(
      lmer(as.formula(paste(fixed, "+", r)), data = data, weights = visits, REML = TRUE,
           control = lctl)))
    if (ok(m)) { cat("  random effects used:", r, "\n"); return(m) }
    cat("  singular or not converged, simplified:", r, "\n")
  }
  cat("  last structure kept although singular\n")
  m
}

fit_glmer <- function(fixed, randoms, data) {
  for (r in randoms) {
    m <- suppressMessages(suppressWarnings(
      glmer(as.formula(paste(fixed, "+", r)), data = data, family = poisson, control = gctl)))
    if (ok(m)) { cat("  random effects used:", r, "\n"); return(m) }
    cat("  singular or not converged, simplified:", r, "\n")
  }
  cat("  last structure kept although singular\n")
  m
}

tidy <- function(ct, label) {
  s <- as.data.frame(summary(ct, infer = c(TRUE, TRUE)))
  lo <- if ("asymp.LCL" %in% names(s)) s$asymp.LCL else s$lower.CL
  hi <- if ("asymp.UCL" %in% names(s)) s$asymp.UCL else s$upper.CL
  s$ratio <- exp(s$estimate)
  s$ratio_low <- exp(lo)
  s$ratio_high <- exp(hi)
  s$index_equivalent <- tanh(s$estimate / 2)
  s$model <- label
  s
}

show <- function(s, by = NULL) {
  cols <- c(by, "contrast", "ratio", "ratio_low", "ratio_high", "index_equivalent", "p.value", "p.holm")
  x <- s[, cols[cols %in% names(s)]]
  num <- sapply(x, is.numeric)
  x[num] <- lapply(x[num], function(v) signif(v, 3))
  print(x, row.names = FALSE)
}

lrt <- function(m_small, m_big, label) {
  a <- suppressMessages(anova(m_small, m_big, refit = TRUE))
  cat(sprintf("  %s: chi2(%d) = %.2f, p = %.4f\n", label, a$Df[2], a$Chisq[2], a$`Pr(>Chisq)`[2]))
}

ee_sc <- list("EE - SC" = c(-1, 1))
results <- list()
for (day in c("day_1", "day_2")) {
  for (which in c("shuffled", "all")) {
    d <- d0[d0$day == day & (which == "all" | d0$block > 1), ]
    tag <- paste(day, if (which == "all") "all blocks" else "blocks 2 to 4")
    cat("\n====================", tag, "====================\n")
    g <- d[d$cls == "grammar", ]

    cat("\n-- staying: log mean visit length, grammar arms --\n")
    gs <- g[which(g$visits > 0 & g$mean_s > 0), ]
    gs$y <- log(gs$mean_s)
    m <- fit_lmer("y ~ environment * tier + block",
                  c("(1 + environment | mouse) + (1 | arm)",
                    "(1 | mouse) + (1 | mouse:environment) + (1 | arm)",
                    "(1 | mouse) + (1 | arm)"), gs)
    lrt(update(m, . ~ . - environment:tier), m, "environment x level")
    s1 <- tidy(contrast(emmeans(m, ~ environment), ee_sc, adjust = "none"), paste(tag, "staying"))
    s2 <- tidy(contrast(emmeans(m, ~ environment | tier), ee_sc, adjust = "none"),
               paste(tag, "staying by level"))
    s2$p.holm <- p.adjust(s2$p.value, "holm")
    cat("  EE against SC, over the levels:\n"); show(s1)
    cat("  by level (p.holm across the three):\n"); show(s2, by = "tier")
    # grammar A against B: counterbalancing crosses grammar with environment, so
    # both can be in one model; a preference for one grammar's sound, whatever
    # the cage it went with, is the grammar term (the environment x group
    # interaction of a model without it)
    mg <- update(m, . ~ . + grammar + grammar:tier)
    lrt(update(mg, . ~ . - grammar - grammar:tier), mg, "grammar A against B, with its interaction with level")
    s3 <- tidy(contrast(emmeans(mg, ~ grammar), list("A - B" = c(-1, 1)), adjust = "none"),
               paste(tag, "staying, grammar A - B"))
    s4 <- tidy(contrast(emmeans(mg, ~ grammar | tier), list("A - B" = c(-1, 1)), adjust = "none"),
               paste(tag, "staying, grammar A - B by level"))
    s4$p.holm <- p.adjust(s4$p.value, "holm")
    cat("  grammar A (ascending) against B (octave alternation), whatever the cage:\n"); show(s3)
    show(s4, by = "tier")
    s5 <- tidy(contrast(emmeans(mg, ~ environment), ee_sc, adjust = "none"),
               paste(tag, "staying, EE - SC with grammar in the model"))
    cat("  EE against SC with grammar in the model:\n"); show(s5)
    results <- c(results, list(s1, s2, s3, s4, s5))

    cat("\n-- entering: entries per arm-block, grammar arms (Poisson) --\n")
    g$obs <- factor(seq_len(nrow(g)))
    mp <- fit_glmer("visits ~ environment * tier + block",
                    c("(1 + environment | mouse) + (1 | arm) + (1 | obs)",
                      "(1 | mouse) + (1 | mouse:environment) + (1 | arm) + (1 | obs)",
                      "(1 | mouse) + (1 | arm) + (1 | obs)"), g)
    e1 <- tidy(contrast(emmeans(mp, ~ environment), ee_sc, adjust = "none"), paste(tag, "entering"))
    e2 <- tidy(contrast(emmeans(mp, ~ environment | tier), ee_sc, adjust = "none"),
               paste(tag, "entering by level"))
    e2$p.holm <- p.adjust(e2$p.value, "holm")
    cat("  EE against SC, over the levels:\n"); show(e1)
    cat("  by level (p.holm across the three):\n"); show(e2, by = "tier")
    mpg <- suppressMessages(suppressWarnings(update(mp, . ~ . + grammar + grammar:tier)))
    e3 <- tidy(contrast(emmeans(mpg, ~ grammar | tier), list("A - B" = c(-1, 1)), adjust = "none"),
               paste(tag, "entering, grammar A - B by level"))
    e3$p.holm <- p.adjust(e3$p.value, "holm")
    cat("  grammar A against B, by level:\n"); show(e3, by = "tier")
    results <- c(results, list(e1, e2, e3))

    cat("\n-- class: silent arm, grammar arms, vocalisation, log mean visit length --\n")
    cs <- d[which(d$visits > 0 & d$mean_s > 0), ]
    cs$y <- log(cs$mean_s)
    cs$cls <- factor(cs$cls, levels = c("silent", "grammar", "vocalisation"))
    mc <- fit_lmer("y ~ cls + block",
                   c("(1 + cls | mouse) + (1 | arm)",
                     "(1 | mouse) + (1 | mouse:cls) + (1 | arm)",
                     "(1 | mouse) + (1 | arm)"), cs)
    c1 <- tidy(contrast(emmeans(mc, ~ cls),
                        list("grammar - silent" = c(-1, 1, 0),
                             "vocalisation - grammar" = c(0, -1, 1),
                             "vocalisation - silent" = c(-1, 0, 1)), adjust = "none"),
               paste(tag, "class"))
    c1$p.holm <- p.adjust(c1$p.value, "holm")
    show(c1)
    results <- c(results, list(c1))
  }
}
# -- session grain: the model of time at the grain of the session-level preference --
# One row per mouse, environment and level: the time in that grammar arm summed
# over the session's four sound blocks.
#   log(1 + session time) ~ environment * level * group + (1 | mouse)
# Each effect is an emmeans contrast averaged over the other factors (the earlier
# analysis fitted the same formula to single arm-blocks and read its
# treatment-coded environment coefficient as the effect in each group and at
# each level).
for (day in c("day_1", "day_2")) {
  cat("\n==================== session grain,", day, "====================\n")
  gb <- d0[d0$day == day & d0$cls == "grammar", ]
  g <- aggregate(time_spent_s ~ mouse + group + environment + tier, data = gb, FUN = sum)
  g$ly <- log1p(g$time_spent_s)
  m <- lmer(ly ~ environment * tier * group + (1 | mouse), data = g, REML = TRUE, control = lctl)
  cat(nrow(g), "mouse x environment x level cells,", length(unique(g$mouse)), "mice\n")
  lrt(update(m, . ~ . - environment:group - environment:tier:group), m,
      "environment x group (with its interaction with level)")
  lrt(update(m, . ~ . - environment:tier - environment:tier:group), m,
      "environment x level (with its interaction with group)")
  show_log <- function(ct, label, by = NULL) {
    s <- as.data.frame(summary(ct, infer = c(TRUE, TRUE)))
    s$model <- paste("session", day, label)
    cols <- c(by, "contrast", "estimate", "lower.CL", "upper.CL", "p.value", "p.holm")
    x <- s[, cols[cols %in% names(s)]]
    num <- sapply(x, is.numeric); x[num] <- lapply(x[num], function(v) signif(v, 3))
    print(x, row.names = FALSE)
    s
  }
  cat("  EE - SC on log(1 + time), averaged over level and group:\n")
  t1 <- show_log(contrast(emmeans(m, ~ environment), ee_sc), "overall")
  cat("  by group, averaged over level:\n")
  t2 <- show_log(contrast(emmeans(m, ~ environment | group), ee_sc), "by group", "group")
  pct <- function(e) sprintf("%+.0f%%", 100 * (exp(e) - 1))
  cat("  the same by group, as percentages more time with the EE grammar (95% interval):\n")
  for (i in seq_len(nrow(t2))) {
    cat(sprintf("    group %s    %s (%s to %s)\n", t2$group[i], pct(t2$estimate[i]),
                pct(t2$lower.CL[i]), pct(t2$upper.CL[i])))
  }
  cat("  by level, averaged over group (p.holm across the three):\n")
  ct3 <- contrast(emmeans(m, ~ environment | tier), ee_sc)
  s3 <- as.data.frame(summary(ct3, infer = c(TRUE, TRUE))); s3$p.holm <- p.adjust(s3$p.value, "holm")
  x <- s3[, c("tier", "contrast", "estimate", "lower.CL", "upper.CL", "p.value", "p.holm")]
  num <- sapply(x, is.numeric); x[num] <- lapply(x[num], function(v) signif(v, 3)); print(x, row.names = FALSE)
  s3$model <- paste("session", day, "by level")
  # The same differences read as percentages of time: a difference d on the log
  # scale means about exp(d) times as much time with the EE grammar (exactly so
  # for 1 + time), that is 100 (exp(d) - 1) percent more.
  pct <- function(e) sprintf("%+.0f%%", 100 * (exp(e) - 1))
  cat("  the same, as percentages more time with the EE grammar (95% interval):\n")
  cat(sprintf("    overall    %s (%s to %s)\n", pct(t1$estimate), pct(t1$lower.CL), pct(t1$upper.CL)))
  for (i in seq_len(nrow(s3))) {
    cat(sprintf("    %-10s %s (%s to %s)\n", s3$tier[i], pct(s3$estimate[i]),
                pct(s3$lower.CL[i]), pct(s3$upper.CL[i])))
  }
  cat("  difference between the groups in EE - SC, at each level:\n")
  t4 <- show_log(contrast(emmeans(m, ~ environment * group | tier), interaction = "pairwise"),
                 "group difference by level", "tier")
  for (s in list(t1, t2, s3, t4)) {
    s$ratio <- NA; s$ratio_low <- NA; s$ratio_high <- NA; s$index_equivalent <- NA
    s$SE <- s$SE; results <- c(results, list(s))
  }
}

sink()
keep <- c("model", "tier", "contrast", "estimate", "SE", "p.value", "p.holm",
          "ratio", "ratio_low", "ratio_high", "index_equivalent")
out <- do.call(rbind, lapply(results, function(s) {
  for (k in keep) if (!k %in% names(s)) s[[k]] <- NA
  s[, keep]
}))
write.csv(out, file.path(args[2], "lmm_contrasts.csv"), row.names = FALSE)
