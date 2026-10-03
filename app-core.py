import numpy as np
import pandas as pd
from shared import (
    app_dir, movie_choices, genres, moods, mood_dict, countries, languages,
    year_min, year_max, runtime_max, df, similarity, pretty,
)
from shiny import App, reactive, render, ui

N_RECS = 4
POSTER_URL = "https://image.tmdb.org/t/p/w342"
FILTERS = ["years", "runtime", "rating", "language", "country", "streaming"]

dark_toggle = ui.input_dark_mode() if hasattr(ui, "input_dark_mode") else None

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "movie", "Want to watch something similar to:",
            multiple=False, selected="", choices=movie_choices, width="100%",
        ),
        ui.input_selectize(
            "genre", "I want to watch these genres:",
            multiple=True, choices=genres, selected=None, width="100%", options={"maxItems": 2},
        ),
        ui.input_selectize(
            "mood", "I'm in this mood:", multiple=False, choices=moods, selected="", width="100%",
        ),
        ui.accordion(
            ui.accordion_panel(
                "More filters",
                ui.input_slider("years", "Release year", year_min, year_max, (year_min, year_max), sep=""),
                ui.input_slider("runtime", "Max runtime (min)", 60, runtime_max, runtime_max),
                ui.input_slider("rating", "Minimum rating", 0, 9, 0, step=0.5),
                ui.input_selectize("language", "Original language", languages, selected="", width="100%"),
                ui.input_selectize("country", "Where do you watch?", countries, selected="IT", width="100%"),
                ui.input_switch("streaming", "Only films on streaming there", False),
            ),
            open=False,
        ),
        ui.div(
            ui.input_action_button("shuffle", "Shuffle", class_="btn-main"),
            ui.input_action_button("surprise", "Surprise me", class_="btn-alt"),
            class_="btn-row",
        ),
        dark_toggle,
        ui.p("Movie data from TMDB.", class_="credit-note"),
        width=320,
    ),
    ui.output_ui("results"),
    ui.include_css(app_dir / "styles.css"),
    title="What should I watch?",
    fillable=False,
)


def pick(pool, n, rng):
    """First call (no shuffle yet) returns the best n; later calls sample from the pool."""
    if rng is None:
        return pool.head(n)
    return pool.sample(n=min(n, len(pool)), random_state=int(rng.integers(1_000_000)))


def movie_card(m, country, heading=None, big=False):
    if m["poster_path"]:
        poster = ui.img(src=POSTER_URL + m["poster_path"], alt=f"Poster of {m['title_pretty']}",
                        class_="poster", loading="lazy")
    else:
        poster = ui.div(m["title_pretty"], class_="poster poster-missing")

    meta = []
    if m["year"]:
        meta.append(str(m["year"]))
    if m["runtime"]:
        meta.append(f"{m['runtime']} min")

    services = m["provider_map"].get(country, [])
    if services:
        stream = ui.div("▶ ", ", ".join(services[:3]), class_="stream stream-yes")
    elif m["provider_map"]:
        stream = ui.div(f"Not on streaming in {countries[country]}", class_="stream stream-no")
    else:
        stream = None

    return ui.card(
        ui.card_header(heading) if heading else None,
        poster,
        ui.div(
            ui.div(m["title_pretty"], class_="movie-title"),
            ui.div(" · ".join(meta), class_="movie-meta"),
            ui.div(f"★ {m['vote_average']:.1f}", ui.span(f" ({int(m['vote_count']):,} votes)"), class_="movie-rating"),
            ui.div(*[ui.span(g, class_="badge-genre") for g in m["genre_list"]], class_="badges"),
            stream,
            ui.p(m["overview"], class_="overview"),
            ui.p(ui.strong("Director: "), pretty(m["director"]), class_="credit") if m["director"] else None,
            ui.p(ui.strong("Cast: "), ", ".join(pretty(c) for c in m["cast_list"]), class_="credit")
            if m["cast_list"] else None,
            class_="card-info",
        ),
        full_screen=True,
        class_="movie-card big" if big else "movie-card",
    )


def server(input, output, session):
    surprise_pick = reactive.value(None)

    @reactive.calc
    def candidates():
        d = df
        lo, hi = input.years()
        d = d[(d["year"] >= lo) & (d["year"] <= hi)]
        d = d[(d["runtime"] <= input.runtime()) & (d["vote_average"] >= input.rating())]
        if input.language():
            d = d[d["original_language"] == input.language()]
        if input.streaming():
            d = d[d["provider_map"].apply(lambda mp: bool(mp.get(input.country())))]

        wanted = list(input.genre() or ())
        if input.mood():
            wanted.append(mood_dict[input.mood()])
        wanted = list(dict.fromkeys(wanted))
        if wanted:
            matches = d["genre_list"].apply(lambda gl: sum(g in gl for g in wanted))
            keep = matches > 0
            d = d[keep].assign(matches=matches[keep])
            # prefer movies that match every requested genre
            if len(d) and (d["matches"] == d["matches"].max()).sum() >= 12:
                d = d[d["matches"] == d["matches"].max()]
        return d

    @reactive.calc
    def recs():
        shuffles = input.shuffle()
        rng = np.random.default_rng(shuffles) if shuffles else None
        title = input.movie()
        if not title and not input.genre() and not input.mood():
            return None

        d = candidates()
        if title:
            idx = int(title)
            d = d.drop(index=idx, errors="ignore")
            if d.empty:
                return d
            sim = similarity(idx)
            d = d.assign(score=sim[d.index.to_numpy()] * (0.8 + 0.2 * d["wr_norm"]))
            return pick(d.nlargest(12, "score"), N_RECS, rng)

        # No reference movie: mix of well-known and lesser-known picks
        popular = d[d["niche"] == 0].nlargest(15, "wr")
        gems = d[d["niche"] == 1].nlargest(30, "wr")
        out = pd.concat([pick(popular, 2, rng), pick(gems, 2, rng)])
        if len(out) < N_RECS:
            rest = d.drop(index=out.index).nlargest(30, "wr")
            out = pd.concat([out, pick(rest, N_RECS - len(out), rng)])
        return out

    @reactive.effect
    @reactive.event(input.surprise)
    def _surprise():
        with reactive.isolate():
            d = candidates()
            if d.empty:
                surprise_pick.set(pd.DataFrame())
                return
            surprise_pick.set(d.nlargest(80, "wr").sample(1))

    @reactive.effect
    @reactive.event(input.movie, input.genre, input.mood, input.shuffle,
                    *[getattr(input, f) for f in FILTERS], ignore_init=True)
    def _clear_surprise():
        surprise_pick.set(None)

    @render.ui
    def results():
        country = input.country()
        s = surprise_pick()
        if s is not None:
            if s.empty:
                return ui.p("No movies match these filters. Try loosening them.", class_="hint")
            return ui.div(movie_card(s.iloc[0], country, "Tonight, watch this", big=True), class_="surprise-wrap")

        r = recs()
        if r is None:
            return ui.p("Pick a movie, a genre or a mood to get started, or press Surprise me.", class_="hint")
        if r.empty:
            return ui.p("No movies match these filters. Try loosening them.", class_="hint")
        names = ["First", "Second", "Third", "Fourth"]
        return ui.layout_columns(
            *[movie_card(r.iloc[i], country, f"{names[i]} recommendation") for i in range(len(r))],
            col_widths={"sm": 12, "md": 6, "xl": 3},
        )


app = App(app_ui, server)
