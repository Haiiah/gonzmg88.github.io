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

{% for bibentry_key in bib_data.entries.keys() %}
<div class="entry">
  <div><span style="font-weight: bold;">{{bib_data.entries[bibentry_key].fields["year"]}}</span>,
    <span style="font-weight: bold;">{{bib_data.entries[bibentry_key].rich_fields.get('title')}}</span>
  </div>
  <div class="author">{{bib_data.entries[bibentry_key].rich_fields.get('author')}}</div>
  {% if "journal" in bib_data.entries[bibentry_key].fields %}
    <div class="journal">{{ bib_data.entries[bibentry_key].rich_fields.get('journal')}}</div>
  {% endif %}
  {% if "booktitle" in bib_data.entries[bibentry_key].fields %}
    <div class="journal">{{ bib_data.entries[bibentry_key].rich_fields.get('booktitle')}}</div>
  {% endif %}
  <div class="blinks">
  {% if "url" in bib_data.entries[bibentry_key].fields %}
        [<a href="{{ bib_data.entries[bibentry_key].fields['url']}}" target="_blank">URL</a>]
  {% endif %}
  {% if "doi" in bib_data.entries[bibentry_key].fields %}
        [<a href="http://doi.org/{{bib_data.entries[bibentry_key].fields['doi'] }}" target="_blank">DOI</a>]
  {% endif %}
  [<a onclick="showFollow(this,'.bibtex')">BibTeX</a>]
  <div class="bibtex">{{extra[bibentry_key]}}</div>
  [<a onclick="showFollow(this,'.abstract')">Abstract</a>]
  <blockquote class="abstract">{{ abstracts[bibentry_key].replace("\n","<br/>")}}</blockquote>
</div>
<div>
{% endfor %}

<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js"></script>
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
