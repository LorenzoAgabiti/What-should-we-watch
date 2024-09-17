import pandas as pd
import random
from shared import app_dir, movie_list, genres, moods, mood_dict, indices, df, niche_df, not_niche_df, cosine_sim1, cosine_sim2
from shiny import App, reactive, ui, render


# Define the app UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "movie",
            "Want to watch something similar to:",
            multiple=False,
            selected=None,
            choices=movie_list,
            width="100%",
        ),

        ui.input_selectize(
            "genre",
            "I want to watch this genres:",
            multiple=True,
            choices=genres,
            selected=None,
            width="100%",
            options=(
                { "maxItems": 2}
            )
        ),
        ui.input_selectize(
            "mood",
            "I'm in this mood:",
            multiple=False,
            choices=moods,
            selected=None,
            width="100%",
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("First recommendation"),
            ui.p("Movie title:"),
            ui.output_text("recommendation1"),
            ui.output_text("time1"),
            ui.card_footer(ui.output_text("overview1")),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Second recommendation"),
            ui.p("Movie title:"),
            ui.output_text("recommendation2"),
            ui.output_text("time2"),
            ui.card_footer(ui.output_text("overview2")),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Third recommendation"),
            ui.p("Movie title:"),
            ui.output_text("recommendation3"),
            ui.output_text("time3"),
            ui.card_footer(ui.output_text("overview3")),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Fourth recommendation"),
            ui.p("Movie title:"),
            ui.output_text("recommendation4"),
            ui.output_text("time4"),
            ui.card_footer(ui.output_text("overview4")),
            full_screen=True,
        ),
        #col_widths=[5]
        col_widths=[3, 3, 3, 3],
    ),
    ui.include_css(app_dir / "styles.css"),
    title="What should I watch?",
    fillable=True,
)


def server(input, output, session):

    @reactive.calc
    @reactive.event(input.movie)
    def get_recommendations():
        title = input.movie()
        if title=='':
            return ['','','','']
        else:
            # Get the index of the movie that matches the title
            idx = indices[title]

            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores1 = list(enumerate(cosine_sim1[idx]))
            sim_scores2 = list(enumerate(cosine_sim2[idx]))

            # Sort the movies based on the similarity scores
            sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
            sim_scores2 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)

            # Get the scores of the 4 most similar movies
            sim_scores1 = sim_scores1[1:5]
            sim_scores2 = sim_scores2[1:5]

            # Get the movie indices
            movie_indices1 = [i[0] for i in sim_scores1]
            movie_indices2 = [i[0] for i in sim_scores2]

            rec1=df['title'].iloc[movie_indices1]
            rec2=df['title'].iloc[movie_indices2]

            rec = pd.concat([rec1,rec2]).drop_duplicates()
            rec = rec.sort_index()
            rec = rec[0:4]
            return rec.values
    
    @reactive.calc
    def get_genre():
        if input.mood():
            gen = mood_dict[input.mood()]
        else:
            gen = list(input.genre())
            #gen = gen.replace(",", '')
            #gen = gen.replace("'", '')
        rec_n=niche_df
        rec_nn=not_niche_df
        rec_n['flag']=0
        rec_nn['flag']=0
        for g in gen:
            #rec_n=niche_df[niche_df['genres'].apply(lambda x: gen in x)]
            rec_n.loc[rec_n['genres'].apply(lambda x: g in x),'flag']=1
            rec_nn.loc[rec_nn['genres'].apply(lambda x: g in x),'flag']=1            
            #rec_nn=not_niche_df[not_niche_df['genres'].apply(lambda x: gen in x)]
        #max_len = min(len(rec_n),len(rec_nn))
        rec_n=rec_n[rec_n['flag']==1]
        rec_nn=rec_nn[rec_nn['flag']==1]
        max_len = min(len(rec_n),len(rec_nn))
        rd_movies=random.sample(range(0, min(40,max_len)), 2)
        rec_n=rec_n.iloc[rd_movies]['title']
        rec_nn=rec_nn.iloc[rd_movies]['title']
        return list(rec_n.values)+list(rec_nn.values)



    @render.text
    def recommendation1():
        rec = get_recommendations()
        if input.genre() or input.mood():
            rec = get_genre()
        return rec[0].upper()
    
    @render.text
    def recommendation2():
       rec = get_recommendations()
       if input.genre() or input.mood():
            rec = get_genre()
       return rec[1].upper()
    
    @render.text
    def recommendation3():
       rec = get_recommendations()
       if input.genre() or input.mood():
            rec = get_genre()
       return rec[2].upper()
    
    @render.text
    def recommendation4():
       rec = get_recommendations()
       if input.genre() or input.mood():
            rec = get_genre()
       return rec[3].upper()
    
    @render.text
    def time1():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title = rec[0]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return f"{df.iloc[idx]['runtime']} minutes"
    @render.text
    def time2():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[1]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return f"{df.iloc[idx]['runtime']} minutes"
    @render.text
    def time3():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[2]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return f"{df.iloc[idx]['runtime']} minutes"
    @render.text
    def time4():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[3]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return f"{df.iloc[idx]['runtime']} minutes"
    @render.text
    def overview1():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title = rec[0]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return df.iloc[idx]['overview']
    @render.text
    def overview2():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[1]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return df.iloc[idx]['overview']
    @render.text
    def overview3():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[2]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return df.iloc[idx]['overview']
    @render.text
    def overview4():
        rec = get_recommendations()
        if input.genre():
            rec = get_genre()
        title =  rec[3]
        if title=='':
            return ''
        else:
            idx = indices[title]
            return df.iloc[idx]['overview']


app = App(app_ui, server)
