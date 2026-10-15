// GeOViLabs — Navbar Mobile Toggle Engine
(function () {
    'use strict';
    
    document.addEventListener('DOMContentLoaded', function () {
        const toggle = document.getElementById('gnav-toggle');
        const linksContainer = document.getElementById('gnav-links');
        const navLinks = document.querySelectorAll('.gnav-link');
        const indicator = document.getElementById('gnav-indicator');
        
        if (!toggle || !linksContainer) return;

        // 1. Fungsi untuk menggerakkan garis underline aktif
        function moveIndicator(activeElement) {
            if (!activeElement || !indicator) return;

            // Hitung posisi link terhadap container (.gnav-links)
            const linkRect = activeElement.getBoundingClientRect();
            const containerRect = linksContainer.getBoundingClientRect();

            // Beri offset opsional agar garis tampak proporsional
            const width = linkRect.width * 0.7; 
            const left = (linkRect.left - containerRect.left) + (linkRect.width * 0.15);

            indicator.style.width = `${width}px`;
            indicator.style.left = `${left}px`;
        }

        // 2. Set item aktif dan gerakkan indikator
        function setActiveLink(targetLink) {
            if (!targetLink) return;

            navLinks.forEach(link => link.classList.remove('gnav-active'));
            targetLink.classList.add('gnav-active');
            moveIndicator(targetLink);
        }

        function closeMenu() {
            linksContainer.classList.remove('open');
            toggle.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }

        // 3. ScrollSpy berbasis posisi scroll
        //    (lebih andal daripada IntersectionObserver: tetap update walau pindah
        //    seksi dengan cepat, dan bisa menangani seksi bersarang seperti #studio
        //    yang membungkus #tools-audio / #tools-video)
        const sections = [];
        navLinks.forEach(link => {
            const sectionId = link.getAttribute('data-section');
            const sectionEl = document.getElementById(sectionId);
            if (sectionEl) {
                sections.push({ id: sectionId, el: sectionEl, link: link });
            }
        });

        let activeSectionId = null;
        let ticking = false;

        function updateActiveSection() {
            const referenceLine = window.innerHeight * 0.28; // garis rujukan ± area tengah-atas
            let current = null;

            sections.forEach(sec => {
                const rect = sec.el.getBoundingClientRect();
                // Seksi dianggap aktif jika puncaknya sudah melewati garis rujukan
                if (rect.top <= referenceLine) {
                    current = sec;
                }
            });

            // Saat sudah di ujung halaman, pastikan seksi terakhir yang aktif
            if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 1) {
                current = sections[sections.length - 1];
            }

            if (current && current.id !== activeSectionId) {
                activeSectionId = current.id;
                setActiveLink(current.link);
            }
        }

        window.addEventListener('scroll', function () {
            if (!ticking) {
                ticking = true;
                requestAnimationFrame(function () {
                    updateActiveSection();
                    ticking = false;
                });
            }
        }, { passive: true });

        // 4. Klik menu navbar: aktifkan link segera (sebelum smooth scroll selesai),
        //    supaya garis bawah selalu muncul ketika navigasi lewat anchor
        navLinks.forEach(link => {
            link.addEventListener('click', function () {
                activeSectionId = link.getAttribute('data-section');
                setActiveLink(link);
            });
        });

        // Initial Position Check pada saat load pertama
        const initialActive = document.querySelector('.gnav-link.gnav-active') || navLinks[0];
        if (initialActive) {
            activeSectionId = initialActive.getAttribute('data-section');
            setActiveLink(initialActive);
        }
        // Koreksi posisi bila halaman dimuat dengan anchor (hash)
        updateActiveSection();

        // Recalculate indicator position on window resize
        window.addEventListener('resize', () => {
            const currentActive = document.querySelector('.gnav-link.gnav-active');
            if (currentActive) moveIndicator(currentActive);
        });

        // 5. Mobile Menu Listeners
        toggle.addEventListener('click', function (e) {
            e.stopPropagation();
            const isOpen = linksContainer.classList.toggle('open');
            toggle.classList.toggle('open', isOpen);
            toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        });

        linksContainer.addEventListener('click', function (e) {
            if (e.target.closest('a')) {
                closeMenu();
            }
        });

        document.addEventListener('click', function (e) {
            if (!toggle.contains(e.target) && !linksContainer.contains(e.target)) {
                closeMenu();
            }
        });
    });
})();