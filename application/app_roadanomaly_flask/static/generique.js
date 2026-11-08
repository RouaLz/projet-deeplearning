// Page standard (objectifs 2 à 6) : image -> predire() -> image résultat + texte.
(() => {
  const img = document.getElementById("img-resultat");
  const texte = document.getElementById("texte-resultat");
  const MESSAGE = `<p class="discret">Déposez une image ou choisissez un exemple, puis cliquez sur « Analyser ».</p>`;

  async function analyser() {
    if (!Entree.source) return;
    Entree.chargement(true);
    try {
      const rep = await Entree.envoyer();
      if (!rep.ok) {
        afficherImage(img, null);
        texte.innerHTML = `<div class="erreur-texte">${echapper(rep.json.erreur || "Erreur inconnue.")}</div>`;
      } else {
        afficherImage(img, rep.json.image);
        texte.innerHTML = rep.json.texte + `<p class="discret">Analyse en ${rep.json.ms} ms.</p>`;
      }
    } catch (e) {
      texte.innerHTML = `<div class="erreur-texte">Le serveur ne répond pas. Vérifiez que « python app.py » tourne toujours.</div>`;
    } finally {
      Entree.chargement(false);
    }
  }

  Entree.bouton.addEventListener("click", analyser);
  Entree.surChangement(analyser);
  Entree.surEffacement(() => { afficherImage(img, null); texte.innerHTML = MESSAGE; });
})();
