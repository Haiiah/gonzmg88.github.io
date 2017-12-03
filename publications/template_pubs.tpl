---
layout: default
title: Publications
---

<style>
.bibtex {
  display:none;
  font-family: 'Courier New', monospace;
  white-space: pre;
  font-size: small;
  background-color: #dedede;
  border:1px solid black;
}

.abstract {
  display:none;
  font-size: small;
  font-style: italic;
  background-color: #dedede;
  border:1px solid black;
}

.author{
  padding-left: 3%;
}
.journal{
  padding-left: 3%;
  font-style: italic;
}
</style>
<h2>Journals</h2>
{% for bibentry_key in ordered_keys %}
{% if "journal" in bib_data.entries[bibentry_key].fields %}
<div class="entry">
  <div><span style="font-style: italic;">{{extra[bibentry_key]["date"].strftime("%h %Y") }}</span>,
    <span style="font-weight: bold;">{{bib_data.entries[bibentry_key].rich_fields.get('title')}}</span>
  </div>
  <div class="author">{{extra[bibentry_key]["authors"]}}</div>
  <div class="journal">{{ bib_data.entries[bibentry_key].rich_fields.get('journal')}}</div>
  <div class="blinks">
  {% if "url" in bib_data.entries[bibentry_key].fields %}
        [<a href="{{ bib_data.entries[bibentry_key].fields['url']}}" target="_blank">URL</a>]
  {% endif %}
  {% if "doi" in bib_data.entries[bibentry_key].fields %}
        [<a href="http://doi.org/{{bib_data.entries[bibentry_key].fields['doi'] }}" target="_blank">DOI</a>]
  {% endif %}
  [<a onclick="showFollow(this,'.bibtex')">BibTeX</a>]
  <div class="bibtex">{{extra[bibentry_key]["bibtex"]}}</div>
  [<a onclick="showFollow(this,'.abstract')">Abstract</a>]
  <blockquote class="abstract">{{ extra[bibentry_key]["abstract"].replace("\n","<br/>")}}</blockquote>
</div>
</div>
{% endif %}
{% endfor %}
<h2>Conferences</h2>
{% for bibentry_key in ordered_keys %}
{% if "booktitle" in bib_data.entries[bibentry_key].fields %}
<div class="entry">
  <div><span style="font-style: italic;">{{extra[bibentry_key]["date"].strftime("%h %Y") }}</span>,
    <span style="font-weight: bold;">{{bib_data.entries[bibentry_key].rich_fields.get('title')}}</span>
  </div>
  <div class="author">{{extra[bibentry_key]["authors"]}}</div>
  <div class="journal">{{ bib_data.entries[bibentry_key].rich_fields.get('booktitle')}}</div>
  <div class="blinks">
  {% if "url" in bib_data.entries[bibentry_key].fields %}
        [<a href="{{ bib_data.entries[bibentry_key].fields['url']}}" target="_blank">URL</a>]
  {% endif %}
  {% if "doi" in bib_data.entries[bibentry_key].fields %}
        [<a href="http://doi.org/{{bib_data.entries[bibentry_key].fields['doi'] }}" target="_blank">DOI</a>]
  {% endif %}
  {% if "slides" in extra[bibentry_key] %}
        [<a href="{{ extra[bibentry_key]['slides']}}" target="_blank">Slides</a>]
  {% endif %}
  {% if "poster" in extra[bibentry_key] %}
        [<a href="{{ extra[bibentry_key]['poster']}}" target="_blank">Poster</a>]
  {% endif %}
  [<a onclick="showFollow(this,'.bibtex')">BibTeX</a>]
  <div class="bibtex">{{extra[bibentry_key]["bibtex"]}}</div>
  [<a onclick="showFollow(this,'.abstract')">Abstract</a>]
  <blockquote class="abstract">{{ extra[bibentry_key]["abstract"].replace("\n","<br/>")}}</blockquote>
</div>
</div>
{% endif %}
{% endfor %}

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/1.12.4/jquery.js"></script>
<script>
		/**/
		function showFollow(event,clase) {
		    var content = $(event).next(clase);
		    content.slideToggle(500,function(){
		        //var caca = content.querySelectorAll('.imagenes_links');
		        //console.log(caca);
		        //wheelzoom(caca);
		    });
		};

</script>
