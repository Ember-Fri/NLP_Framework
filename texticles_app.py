from texticles import Textastic

def main():
    tt = Textastic()

    # Load stopwords
    tt.load_stop_words('stopwords.txt')

    # Load text files into the framework

    # PRE CHERNOBYL
    tt.load_text('A Building Boom for Nuclear Power Plants.txt', 'Boom1972', category="Before Chernobyl")
    tt.load_text('Aiken Says Giant Utilities Seek To Monopolize Nuclear Power.txt', 'Aiken1971',
                 category="Before Chernobyl")
    tt.load_text('Is Nuclear Too Costly_.txt', 'Costly1975', category="Before Chernobyl")
    tt.load_text('Nuclear Power Arrives—Again.txt', 'Arrives1970', category="Before Chernobyl")
    tt.load_text('The Maturity and Future of Nuclear Energy.docx.txt', 'Future1976', category="Before Chernobyl")

    # POST CHERNOBYL
    tt.load_text('Here_s how nuclear energy production has changed since 1965.txt', 'Changes2021',category="After Chernobyl")
    tt.load_text('Hungry for Energy, Amazon, Google and Microsoft Turn to Nuclear Power.txt', 'TechNuclear2022',
                 category="After Chernobyl")
    tt.load_text('PART I AP IMPACT US nuke regulators weaken safety rules.txt', 'Safety2022',
                 category="After Chernobyl")
    tt.load_text('Three decades of nuclear safety.docx.txt', 'Decades2016', category="After Chernobyl")
    tt.load_text('Three Mile Island, and Nuclear Hopes and Fears.txt', 'TMI2000', category="After Chernobyl")


    # Create the Sankey diagram
    tt.wordcount_sankey(k=5)

    # Initial polarity and subjectivity visualizations
    tt.subjectivity_polarity_viz()
    # check sentiment analysis
    print(tt.data['polarity'])
    print(tt.data['subjectivity'])

    #cosine similarity
    tt.cos_similarity_plt()


if __name__ == "__main__":
    main()