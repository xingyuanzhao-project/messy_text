(function (w,d,t) {
    _ml = w._ml || {};
    _ml.eid = '90950';
    var s, cd, tag; s = d.getElementsByTagName(t)[0]; cd = new Date();
    tag = d.createElement(t); tag.async = 1;
    tag.src = 'https://ml314.com/tag.aspx?' + cd.getDate() + cd.getMonth();
    s.parentNode.insertBefore(tag, s);
    console.debug("Bombora Visitor Tag: Initialized");
})(window,document,'script');

window.addEventListener("load", function bomboraTrackLoad(){
    _ml.q = _ml.q || [];
    _ml.q.push(['track']);
    console.debug("Bombora Visitor Tag: Page load tracked");
});