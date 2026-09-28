use mtg_kernel::card_def::{
    card_id_by_name, preflight_fully_supported_deck, CardCapability, CardType, DynamicCountDef,
    Subtype, CARD_DEFS,
};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::{Counters, GameObject, GameState, ObjectStateV4, Step, Zone};

fn ready_main1() -> GameState {
    let mut state = GameState::new_from_libraries(&[], &[], |_| String::new(), 43);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state
}

fn put_object(state: &mut GameState, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"));
    let id = state.objects.push(GameObject {
        card_def,
        name: name.to_string(),
        owner: PlayerId::P0,
        controller: PlayerId::P0,
        zone,
        tapped: false,
        summoning_sick: zone == Zone::Battlefield,
        damage: 0,
        counters: Counters::default(),
        attachments: Vec::new(),
        v4: ObjectStateV4::from_card_def(card_def),
        spell_copy_origin: None,
        plotted_turn: None,
        zone_change_count: 0,
    });
    match zone {
        Zone::Hand => state.players[0].hand.push(id),
        Zone::Battlefield => state.players[0].battlefield.push(id),
        Zone::Graveyard => state.players[0].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Library => state.players[0].library.push(id),
        Zone::Command => state.command.push(id),
        Zone::Stack => panic!("test helper does not fabricate stack items"),
    }
    id
}

fn assert_castable(state: &mut GameState, spell: ObjectId, expected: bool) {
    match engine::advance_until_decision(state) {
        Decision::CastSpellOrPass {
            castable_spells, ..
        } => assert_eq!(castable_spells.contains(&spell), expected),
        other => panic!("expected CastSpellOrPass, got {other:?}"),
    }
}

fn resolve_spell(state: &mut GameState, spell: ObjectId) {
    for _ in 0..8 {
        if state.objects.get(spell).zone == Zone::Graveyard {
            return;
        }
        match engine::advance_until_decision(state) {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision while resolving spell: {other:?}"),
        }
    }
    panic!("spell did not resolve within the bounded priority walk");
}

#[test]
fn azorius_lands_have_exact_types_mana_and_entry_contracts() {
    let expected = [
        ("Azorius Guildgate", &[Subtype::Gate][..]),
        (
            "Idyllic Beachfront",
            &[Subtype::Plains, Subtype::Island][..],
        ),
    ];
    let ids: Vec<u16> = expected
        .iter()
        .map(|(name, subtypes)| {
            let id = card_id_by_name(name).unwrap();
            let def = &CARD_DEFS[id as usize];
            assert_eq!(def.capability, CardCapability::Full, "{name}");
            assert!(def.has_type(CardType::Land), "{name}");
            assert_eq!(def.subtypes, *subtypes, "{name}");
            assert_eq!(
                def.mana_ability_choices,
                &[ManaColor::W, ManaColor::U],
                "{name}"
            );
            assert!(def.enters_battlefield_tapped, "{name}");
            id
        })
        .collect();
    preflight_fully_supported_deck(&ids).unwrap();

    for (name, _) in expected {
        let mut state = ready_main1();
        let land = put_object(&mut state, name, Zone::Hand);
        engine::step(&mut state, Action::PlayLand(land)).unwrap();
        assert_eq!(state.objects.get(land).zone, Zone::Battlefield, "{name}");
        assert!(state.objects.get(land).tapped, "{name}");

        state.objects.get_mut(land).tapped = false;
        engine::step(
            &mut state,
            Action::ActivateManaAbilityChoice(land, ManaColor::U),
        )
        .unwrap();
        assert!(state.objects.get(land).tapped, "{name}");
        assert_eq!(state.players[0].mana_pool[ManaColor::U.pool_index()], 1);
    }
}

#[test]
fn of_one_mind_costs_one_only_with_human_and_non_human_creatures() {
    let mut human_only = ready_main1();
    human_only.players[0].mana_pool[ManaColor::U.pool_index()] = 1;
    let spell = put_object(&mut human_only, "Of One Mind", Zone::Hand);
    put_object(&mut human_only, "Outlaw Medic", Zone::Battlefield);
    assert_castable(&mut human_only, spell, false);

    put_object(&mut human_only, "Faerie Seer", Zone::Battlefield);
    assert_castable(&mut human_only, spell, true);

    let mut non_human_only = ready_main1();
    non_human_only.players[0].mana_pool[ManaColor::U.pool_index()] = 1;
    let spell = put_object(&mut non_human_only, "Of One Mind", Zone::Hand);
    put_object(&mut non_human_only, "Faerie Seer", Zone::Battlefield);
    put_object(&mut non_human_only, "Faerie Miscreant", Zone::Battlefield);
    assert_castable(&mut non_human_only, spell, false);
}

#[test]
fn of_one_mind_pays_one_blue_and_draws_exactly_two() {
    let def = &CARD_DEFS[card_id_by_name("Of One Mind").unwrap() as usize];
    let reducer = def.generic_cost_reduction.expect("conditional reducer");
    assert_eq!(reducer.generic_per_count, 2);
    assert_eq!(
        reducer.count,
        DynamicCountDef::ControllerHasCreatureWithAndWithoutSubtype(Subtype::Human)
    );

    let mut state = ready_main1();
    state.players[0].mana_pool[ManaColor::U.pool_index()] = 1;
    put_object(&mut state, "Outlaw Medic", Zone::Battlefield);
    put_object(&mut state, "Faerie Seer", Zone::Battlefield);
    put_object(&mut state, "Mountain", Zone::Library);
    put_object(&mut state, "Island", Zone::Library);
    let spell = put_object(&mut state, "Of One Mind", Zone::Hand);

    assert_castable(&mut state, spell, true);
    engine::step(&mut state, Action::CastSpell(spell)).unwrap();
    resolve_spell(&mut state, spell);

    assert_eq!(state.players[0].mana_pool, [0; 6]);
    assert_eq!(state.objects.get(spell).zone, Zone::Graveyard);
    assert_eq!(state.players[0].hand.len(), 2);
    assert!(state.players[0].library.is_empty());
}
